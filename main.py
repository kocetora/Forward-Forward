import torch
import time
from utils import *
from model import *
from sklearn.metrics import confusion_matrix
    
if __name__ == "__main__":
    torch.manual_seed(42)
    train_loader, eval_train_loader, eval_test_loader = CIFAR10_loaders()
    net = FFNet([3072, 3072, 3072, 10])

    time_training_start = time.time()
    net.train_3(train_loader)
    time_training_end = time.time()
    training_time = round(time_training_end - time_training_start, 2)

    print(f"Training time: {training_time}s")

    print('train error:', str(round((1.0 - net.predict(eval_train_loader)) * 100, 2)) + '%')

    print('test error:', str(round((1.0 - net.predict(eval_test_loader)) * 100, 2)) + '%')

    def iterate_cm(dims, num_epochs, pow = 4, norm = 3, cm = None, model_path = None):
        net = FFNet(dims, pow, norm)  
        if(model_path):
          net.load_state_dict(torch.load(model_path))
        time_training_start = time.time()
        net.train(train_loader, num_epochs, cm)
        time_training_end = time.time()
        training_time = time_training_end - time_training_start
        train_accuracy, predicted_y_train, y_train = net.predict(eval_train_loader)
        test_accuracy, predicted_y_test, y_test = net.predict(eval_test_loader)
        cm = confusion_matrix(y_test.tolist(), predicted_y_test.tolist())

        # print("Total parameters:", net.count_neurons())
        return round(training_time, 2), round(train_accuracy * 100, 2), round(test_accuracy * 100, 2), cm, net

    def cm_experiment(
        dims, 
        num_epochs = 60, 
        iters = 3, 
        pow = 2, 
        norm = 2, 
        model_paths = [], 
        cm=[], 
        times=[], 
        train_accuracies=[],
        test_accuracies=[]
        ):
      model_path = None  
      for i in range(1, iters + 1):
        print("Epochs:",(i-1)*num_epochs, "-", i*num_epochs)
        if(len(cm)): 
          cm = clarify_cm(cm)
        if(len(model_paths)): 
          model_path = model_paths[-1]

        training_time, train_accuracy, test_accuracy, cm, net = iterate_cm(dims, num_epochs, pow, norm, cm, model_path)
        if(type(cm) == 'list'):
          cm = np.array(cm)

        file_path = f"modelname_{i}.pth"
        torch.save(net.state_dict(), file_path)
        model_paths.append(file_path)

        times.append(time)
        train_accuracies.append(time)
        test_accuracies.append(time)

        plt_cm(cm)
        print("Time", training_time, "s")
        print("Train accuracy", train_accuracy, "%")
        print("Test accuracy", test_accuracy, "%")
      return model_paths, train_accuracies, test_accuracies, times, cm
    
    cm_experiment(dims=[3072, 3072, 3072], num_epochs = 10, iters = 6)

    def experiment(dims, epochs = 90, steps=15, pow = 2, norm = 2, iters = 3):
        times = []
        train_accuracies = []
        test_accuracies = []

        # stages = epochs / steps
        for i in range(iters):
            net = FFNet(dims, pow, norm)  
            time_training_start = time.time()
            net.train(train_loader, epochs, cm=[])
            time_training_end = time.time()
            training_time = time_training_end - time_training_start
            times.append(training_time)
            train_accuracy, predicted_y_train, y_train = net.predict(eval_train_loader)
            test_accuracy, predicted_y_test, y_test = net.predict(eval_test_loader)
            train_accuracies.append(train_accuracy * 100)
            test_accuracies.append(test_accuracy * 100)

        print("Total parameters:", net.count_neurons())
        meanWithStdDeviation("Time", times, "s")
        meanWithStdDeviation("Train accuracy", train_accuracies, "%")
        meanWithStdDeviation("Test accuracy", test_accuracies, "%")

        plt_cm(confusion_matrix(y_test.tolist(), predicted_y_test.tolist()))
    experiment(dims=[3072, 3072, 3072], epochs = 20, iters = 3)
