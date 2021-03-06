# Character LSTM
# Builds a model and trains on text

import theano
import theano.tensor as T
from theano_lstm import *
import numpy as np
import sys
import os
import re
import pickle
from datetime import datetime


class CharLSTM:


    def __init__(self, layers, num_possible_characters):
        print("Building the model...")
        self.rng = theano.tensor.shared_randomstreams.RandomStreams()

        self.model = StackedCells(num_possible_characters, layers=layers, activation=T.tanh, celltype=LSTM)
        self.model.layers[0].in_gate2.activation = lambda x: x
        self.model.layers.append(Layer(layers[-1], num_possible_characters, lambda x: T.nnet.softmax(x)[0]))

        num_steps = T.scalar(dtype='int32')
        # function to put into scan to fire the network recurrently
        def step(prev_char, *prev_hiddens):
            new_hiddens = self.model.forward(int_to_onehot(T.cast(prev_char, 'int32'), num_possible_characters), prev_hiddens)
            dist = new_hiddens[-1]
            next_char = self.rng.choice(size=[1], a=num_possible_characters, p=dist)
            return [T.cast(next_char, 'int32')] + new_hiddens[:-1]

        results, updates = theano.scan(step, n_steps=num_steps,
        outputs_info=[dict(initial=np.int32([-1]),taps=[-1])] 
        + [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')])

        self.forward_pass = theano.function([num_steps], [results[0].dimshuffle((1,0))[0]], updates=updates, allow_input_downcast=True)

        training_data = T.matrix('training data') # list of character values less than num_possible_characters

        def step_inner(prev_char, desired_output, *prev_hiddens):
            new_hiddens = self.model.forward(int_to_onehot(prev_char, num_possible_characters), prev_hiddens)
            prob_correct = new_hiddens[-1][desired_output]
            return [prob_correct] + new_hiddens[:-1]

        # I have no idea whether nesting scan will work at all
        def step_outer(training_sample, *initial_states):
            print(list(initial_states))
            # different call to scan that uses the training data as prior timesteps
            results_inner, updates_inner = theano.scan(step_inner, n_steps=training_sample.shape[0], sequences=[dict(input=T.cast(T.concatenate(([0], training_sample)), 'int32'), taps=[0,1])],
            outputs_info=[None] + list(initial_states))
            return results_inner, updates_inner

        results_outer, updates_outer = theano.scan(step_outer, n_steps=training_data.shape[0], 
            sequences=[training_data], 
            non_sequences=[layer.initial_hidden_state for layer in self.model.layers if hasattr(layer, 'initial_hidden_state')],
            outputs_info=[None, None])

        results_inner = results[0] # this should be a list of each "results" from step_inner
        updates_inner = results[1] # this should be a list of updates

        # I want to find the zero position of each results vector in results_inner

        prob_correct_v = results_inner[:][0] # should be a matrix of probabilities between 0 and 1
        cost = -T.mean(T.log(prob_correct_v)) # mean should take the average across all dimensions

        u, gsums, xsums, lr, max_norm = create_optimization_updates(cost, self.model.params, method='adadelta')

        # combine all the updates into one dictionary
        all_updates = {}
        for d in updates_inner:
            all_updates.update(d)
        all_updates.update(updates_outer)

        self.training_pass = theano.function([training_data], [cost], updates=all_updates + u, allow_input_downcast=True)
        self.validation_pass = theano.function([training_data], [cost], updates=all_updates, allow_input_downcast=True)

    def validate(self, validation_set, minibatch_size):
        examples_to_train_on = np.random.choice(len(validation_set), minibatch_size, replace=False)
        cost = 0
        for sample in validation_set[examples_to_train_on]:
            cost += self.validation_pass(sample)[0]
        return cost/minibatch_size


    # train the model on a numpy 2d array of character sequences
    def train(self, dataset, minibatch_size=100, max_num_batches=5000, output_dir="output", record_weights=True, record_samples=False):
        if record_weights or record_samples:
            os.mkdir(output_dir)
        print("Training...")
        num_training_examples = (len(dataset) * 3) // 4
        num_validation_examples = len(dataset) - num_training_examples
        prev_cost = sys.maxsize
        strikes=0
        for batch in range(1, max_num_batches):
            examples_to_train_on = np.random.choice(num_training_examples, minibatch_size, replace=False)
            cost = 0
            for sample in dataset[examples_to_train_on]:
                cost = self.training_pass(sample)
            if batch % 10 == 0:
                print("Minibatch ", batch * minibatch_size / num_training_examples)
                print("Loss: ", cost)
                

            if batch % 10 == 0:
                print("Validating...")
                if record_weights:
                    params = []
                    for p in self.model.params:
                        params += [p.get_value()]
                    pickle.dump(params, open(output_dir + '/batch' + str(batch) + '.p', 'wb'))
                if record_samples:
                    output_file = open(output_dir + '/batch' + str(batch) + '.txt', 'w')
                print("Average Error: ", self.validate(dataset[num_training_examples:], minibatch_size))
                




def int_to_onehot(n, len):
    a = T.zeros([len])
    if n == -1: return a
    a = T.set_subtensor(a[n], 1)
    return a

# filepath is the path to the text file to parse
# delim is a regex to split the file into training examples by
def parse_text_file(filepath, delim='\n'):
    f = open(filepath)
    raw_text = f.read()
    letters = set(raw_text)
    letter_number_dict = dict([(elem, i) for elem, i in zip(letters, range(len(letters)))])
    lines = []
    for line in re.split(delim, raw_text):
        if line is not '':
            lines += [np.array([letter_number_dict[l] for l in line])]
    return np.array(lines), letter_number_dict

def output_to_str(data, letter_number_dict):
    number_letter_dict = dict((v, k) for k, v in letter_number_dict.items())
    s = ''
    for d in data:
        if d == len(letter_number_dict): return s
        s += number_letter_dict[d]
    return s

if __name__ == '__main__':
    data, lnd = parse_text_file(sys.argv[1], delim='\n|\.')
    for i in range(len(data)):
        data[i] = np.append(data[i], len(lnd))
    m = CharLSTM([100], len(lnd)+1)
    m.train(data)
    output_file = open('output' + datetime.now().strftime('%m-%d %H:%M'), 'w')
    for i in range(10):
        output_file.write(output_to_str(m.forward_pass(1000)[0], lnd) + '\n')
