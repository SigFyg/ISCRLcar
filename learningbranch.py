import numpy as np
import random
import csv
from nn import neural_net, LossHistory
import os.path
import timeit


NUM_INPUT = 3
GAMMA = 0.9  # Forgetting.
TUNING = False  # If False, just use arbitrary, pre-selected params.


def train_net(model, params, old_state, state, t, epsilon, replay, loss_log,
              car_distance, data_collect, max_car_distance, fps):
    filename = params_to_filename(params)
    observe = 300
    train_frames = 3000
    buffer = params['buffer']
    batchSize = params['batchSize']
    # Let's time this. 
    start_time = timeit.default_timer()
    old_state = np.array(old_state)
    state = np.array(state)
    # Run the frames.
    if t < train_frames:
        print(t)
        # Choose an action.
        if random.random() < epsilon or t < observe:
            action = str(np.random.randint(0, 3))  # random
        else:
            # Get Q values for each action.
            qval = model.predict(np.array([old_state]), batch_size=1)
            action = str(np.argmax(qval))  # best


        # (Don't) Take action, observe new state and get our treat.
        reward = 0
        if state[0] < 13.0 or state[1] < 13.0 or state[2] < 13.0:# check if crashed
            reward = -500
        else:
            # Higher readings are better, so return the sum.
            reward = -5 + int((state[0] + state[1] + state[2]) / 10)

        if action=='2': #if the car moves forward
            car_distance+=10

        # Experience replay storage.
        replay.append((old_state, int(action), reward, state))
        # If we're done observing, start training.
        if t > observe:
            # If we've stored enough in our buffer, pop the oldest.
            #Keeps the replay size at 50000
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            # Get training values.
            X_train, y_train = process_minibatch(minibatch, model)

            # Train the model on this batch.
            history = LossHistory()
            model.fit(
                X_train, y_train, batch_size=batchSize,
                nb_epoch=1, verbose=0, callbacks=[history]
            )
            loss_log.append(history.losses)
        # Update the starting state with S'.
        old_state = state

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1/train_frames)

        # We died, so update stuff.
        if reward == -500:
            action = '3'
            # Log the car's distance at this T.
            data_collect.append([t, car_distance])
           
            # Update max.
            if car_distance > max_car_distance:
                 max_car_distance = car_distance

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time

            # Output some stuff so we can watch.
            print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" %
                  (max_car_distance, t, epsilon, car_distance, fps))

            # Reset.
            car_distance = 0
            #start_time = timeit.default_timer()

        # Save the model every 2,000 frames.rm
        if t % 50 == 0:
            model.save_weights('saved-models/' + filename + '-' +
                               str(t) + '.h5',
                               overwrite=True)
            print("Saving model %s - %d" % (filename, t))

    # Log results after we're done all frames.
    else:
        action = 'KILL'
    log_results(filename, data_collect, loss_log, replay)
    return action, model, old_state, state, epsilon, replay, loss_log, car_distance, data_collect, max_car_distance, fps

def log_results(filename, data_collect, loss_log, replay):
    # Save the results to a file so we can graph it later.
    with open('results2/learn_data-' + filename + '.csv', 'w') as data_dump:#if it's not write, it will repeat itself
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open('results2/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

    with open('results2/replay-' + filename + '.csv', 'w') as replay_save:
        wr = csv.writer(replay_save)
        for element in replay:
            wr.writerow(element[0])
            wr.writerow([element[1]])
            wr.writerow([element[2]])
            wr.writerow(element[3])



def process_minibatch(minibatch, model):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        #np.reshape(old_state_m,(1,3))
        old_qval = model.predict(np.array([old_state_m]), batch_size=1)
        # Get prediction on new state.
        #np.reshape(new_state_m,(1,3))
        newQ = model.predict(np.array([new_state_m]), batch_size=1)
        # Get our best move. I think?
        maxQ = np.max(newQ)
        y = np.zeros((1, 3))
        y[:] = old_qval[:]
        # Check for terminal state.
        if reward_m != -500:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
        # Update the value for the action we took.
        
        y[0][int(action_m)] = update
        X_train.append(np.array([old_state_m]).reshape(NUM_INPUT,))
        y_train.append(y.reshape(3,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])


def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results2/loss_data-' + filename + '.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results2/loss_data-' + filename + '.csv', 'a').close()
        print("Starting test.")
        # Train.
        model = neural_net(NUM_INPUT, params['nn'])
        train_net(model, params)
    else:
        print("Already tested.")


if __name__ == "__main__":
    if TUNING:
        param_list = []
        nn_params = [[164, 150], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100, 400]
        buffers = [10000, 50000]

        for nn_param in nn_params:
            for batchSize in batchSizes:
                for buffer in buffers:
                    params = {
                        "batchSize": batchSize,
                        "buffer": buffer,
                        "nn": nn_param
                    }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)

    else:
        nn_param = [64, 50]
        params = {
            "batchSize": 100,
            "buffer": 50000,
            "nn": nn_param
        }
        model = neural_net(NUM_INPUT, nn_param)# model is a network with 3 layers
        train_net(model, params)


