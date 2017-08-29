from __future__ import division
from numpy import genfromtxt
from random import sample
import matplotlib.pyplot as plt
import sys

# Loading the data and definig the attributes and classes.
cardata = genfromtxt('car.data', dtype=None, delimiter=',')
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
classes = ['unacc', 'acc', 'good', 'vgood']

def sample_the_data(data):
    ''' Takes the data array as an argument and returns randomly third of the data as 
    a training sample and the rest two thirds as a testing sample '''
    training_indexes = sample(range(len(data)), int(round(len(data)/3)))
    training_sample = data[training_indexes]
    testing_indexes = [i for i in range(len(data)) if i not in training_indexes]
    testing_sample = data[testing_indexes]
    return training_sample, testing_sample
    
def NBC_train(training_sample, attributes, classes):
    ''' The training function takes as arguments the training sample, the attributes names and the classes and returns the model of attributes values frequencies and the classes count '''
    classes_count = [0 for x in classes]
    model = [[{} for k in range(len(attributes))] for i in range(len(classes))]
    for instance in training_sample:
        classes_count[classes.index(instance[-1])] +=1
        for j in range(len(instance)-1):
            key = instance[j]
            d = model[classes.index(instance[-1])][j]
            if key in d:
                d[key] += 1
            else:
                d[key] = 1
    return [model, classes_count]
                
def NBC_classify(instance, model, classes):
    ''' the classify function takes as argument one new instance, the model and the classes list and returns the most probable class classification for that instance '''
    classes_count = model[1]
    overall_count = sum(classes_count)
    potentials = [0 for x in model[1]]
    model = model[0]
    for i in range(len(classes_count)):
        att_probs = []
        att_probs.append(classes_count[i]/overall_count)
        for j in range(len(instance)-1):
            try:
                value_count = model[i][j][instance[j]]
            except KeyError:
                value_count = 0
            att_probs.append((value_count+1)/(classes_count[i]+len(classes)))
        potentials[i] = reduce(lambda x, y: x*y, att_probs)
    max_index = potentials.index(max(potentials))
    return classes[max_index]
            
def error(testing_sample, model, classes):
    '''the error function takes as an argument the testing sample, the model and
    the classes values. it classify every instance and compare the result with the 
    actual classification. it returns an error ration, a list of the actual 
    classifications and a list of the predicted classifications. '''
    errors = 0
    actual = []
    predicted = []
    for instance in testing_sample:
        c = NBC_classify(instance, model, classes)
        if c != instance[-1]:
            errors +=1
        actual.append(instance[-1])
        predicted.append(c)
    return [(errors/len(testing_sample))*100, actual, predicted]
                 
def test(K):
    '''the test function takes an integer K and do K experiments. in each experiment 
    it takes a new random training and testing samples and do the training, classification and error calculation. it returns a list of all error ratios
    and the mean error ratio '''
    error_ratios = []
    for i in range(K):        
        training_sample, testing_sample = sample_the_data(cardata)
        model = NBC_train(training_sample, attributes, classes)
        error_ratio = error(testing_sample, model, classes)[0]
        error_ratios.append(error_ratio)
        sys.stdout.write('Doing Experiment '+str(i+1)+'\r')
        sys.stdout.flush()
    print '\n'
    return [sum(error_ratios)/max(len(error_ratios), 1), error_ratios]


def plot_error_histogram(error_ratios):
    '''it takes a list of error ratios and plots a histogram of the frequency of each error ratio'''
    plt.hist(error_ratios, 20, facecolor='green')
    plt.title("Error Histogram")
    plt.xlabel("Error_ratio")
    plt.ylabel("Frequency")

def confusion_matrix(actual, predicted, classes):
    '''this function takes a list of actual classifications, another list of predicted
     classifications and a list of the classes. it calculates the confusion matrix, 
     prints it to the screen, plots it as a color map and returns it as a list of lists'''
    matrix = [[0 for i in classes] for x in range(len(classes))]
    actual = [classes.index(i) for i in actual]
    predicted = [classes.index(i) for i in predicted]
    for i,j in zip(actual, predicted):
        matrix[i][j] +=1
    
    print 'Printing confusion Matrix'
    print '========================='
    print 'predicted',
    print ' '.join(classes)   
    print 'actual'
    for i in range(len(classes)):
        print classes[i], '     ','    '.join([str(w) for w in matrix[i]]) 
        
    plt.matshow(matrix, cmap=plt.cm.hot_r)
    plt.title('Confusion Matrix plot')
    plt.colorbar()
    plt.ylabel('actual class')
    plt.xlabel('predicted class')  
    plt.xticks(range(len(classes)), tuple(classes))
    plt.yticks(range(len(classes)), tuple(classes))
    plt.show() 
    
    return matrix

# defining how many experiments we want to conduct.
K = 100
# doing the experiments.
mean_error, error_ratios = test(K)
# printing the mean error rate to the screen.
print 'Mean error after', K, 'random experiments is',round(mean_error,2), '%', '\n'
# plotting the error histogram.
plot_error_histogram(error_ratios)
# doing one more experiment for sake of computing the confusion matrix and plotting it.    
training_sample, testing_sample = sample_the_data(cardata)
model = NBC_train(training_sample, attributes, classes)
error_ratio, actual, predicted = error(testing_sample, model, classes)    
m = confusion_matrix(actual, predicted, classes)


    
    
    
    
    
    