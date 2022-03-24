from typing import Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from Enums import SNR
from Parameters import Parameters

from tf.keras.callbacks import History


class Evaluation:
    """
    Evaluation class performing all evaluation needed for the tensorflow models.
    Includes printing and saving the training history, as well as confusion matrices.
    """
    def __init__(self, model, model_path: str, history: History, x_test: np.ndarray,
                 y_test: np.ndarray, snr: SNR, parameter: Parameters, model_name: str,
                 best_weights: str = '', smooth_range: int = 2, save_figures: bool = True,
                 percentage_conf_matrix: bool = False) -> None:
        """
        Initialise the class.
        Will load the best model weights (if given) and run the model evaluation.

        Args:
            model: Trained Tensorflow model
            model_path: Path where graphs should be saved
            history: Training history object containing the training and evaluation accuracy and loss.
            x_test: Test dataset
            y_test: Test labels
            snr: SNR level on which the model was trained
            parameter: Parameters object containing information about the feature extraction
            best_weights: String containing the path to the best weights. If not given, use current weights instead.
            smooth_range: The range which should be used for post_processing.
            save_figures: Boolean whether the plots should be saved or not. They will be saved to model_path.
            percentage_conf_matrix: If the confusion matrix should be row normalized or not.
        """
        self.model = model  # tensorflow model
        self.model_path = model_path
        self.history = history
        self.x_test = x_test
        self.y_test = y_test
        self.test_loss, self.test_acc = 0, 0  # created in start function
        self.correct_labels = []  # reshaped and argmax of y_test
        self.predicted_labels = []  # reshaped and argmax of predictions
        self.predicted_labels_post_processed = []  # placeholder for smoothened predicted labels

        # placeholders for the confusion matrices
        self._conf_matrix: np.ndarray = np.array([])
        self._conf_matrix_post_processed: np.ndarray = np.array([])

        self.snr = snr
        self.parameter = parameter
        self.best_weights = best_weights  # path to the best model

        self.smooth_range = smooth_range
        self.save_figures = save_figures
        self.percentage = percentage_conf_matrix

        # predict the results
        self.start()

    def plot_acc(self):
        """
        Plot the accuracy from the given model
        """
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if self.save_figures:
            plt.savefig(self.model_path + '/accuracy.png')
        plt.show()

    def plot_loss(self):
        """
        Plot the loss from the given model
        """
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model losss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if self.save_figures:
            plt.savefig(self.model_path + '/loss.png')
        plt.show()

    def create_predictions(self):
        """
        Predict the outcomes for the test set
        """
        predictions = self.model.predict(self.x_test)
        predictions_reshaped = predictions.reshape(-1, 2)
        correct_reshaped = self.y_test.reshape(-1, 2)
        self.predicted_labels = np.argmax(predictions_reshaped, axis=1)
        self.correct_labels = np.argmax(correct_reshaped, axis=1)

    def conf_matrix(self, post_processing=False):
        """
        Create a confusion matrix
        """
        if not post_processing:
            predicted_labels = self.predicted_labels
        else:
            predicted_labels = self.predicted_labels_post_processed
        cf_matrix = confusion_matrix(self.correct_labels, predicted_labels)
        if self.percentage:
            # transpose matrix to get the correct division, then transpose it back to the correct orientation
            cf_matrix = (cf_matrix.T / np.sum(cf_matrix, axis=1)).T
        print(cf_matrix)
        plt.figure(figsize=(16, 9))
        sns.heatmap(cf_matrix, annot=True, xticklabels=['no speech', 'speech'],
                    yticklabels=['no speech', 'speech'], fmt='g')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if self.save_figures:
            if not post_processing:
                plt.savefig(self.model_path + '/conf_matrix_no_post.png')
            else:
                plt.savefig(self.model_path + '/conf_matrix_post.png')
        plt.show()
        return cf_matrix

    def calculate_hter(self, post_processing=False):
        """
        Calculate the HTER, MR and FAR for the confusion matrix
        """
        if not post_processing:
            cf_matrix = self._conf_matrix
        else:
            cf_matrix = self._conf_matrix_post_processed
        mr = cf_matrix[1][0]/(sum(cf_matrix[1])) * 100
        far = cf_matrix[0][1]/(sum(cf_matrix[0])) * 100
        hter = (mr+far)/2
        if not post_processing:
            hter_string = f'The HTER is {hter:.3f}% for SNR {self.snr.value}, ' \
                          f'with MR being {mr:.3f}% and FAR being {far:.3f}%'
        else:
            hter_string = f'The post-processing HTER is {hter:.3f}% for SNR {self.snr.value}, ' \
                          f'with MR being {mr:.3f}% and FAR being {far:.3f}%'
        print(hter_string)

    def smoothen_specs(self):
        """
        Apply smoothing to the results
        """
        predicted_post_processing = []
        for pred_idx in range(len(self.predicted_labels)):
            start = max(pred_idx-self.smooth_range, 0)
            # + smoothing range + 1 as last element is not included in slice
            end = min(len(self.predicted_labels), pred_idx+self.smooth_range+1)
            avg = np.average(self.predicted_labels[start:end])
            prediction = int(np.rint(avg))  # round to either 0 or 1
            predicted_post_processing.append(prediction)
        return np.array(predicted_post_processing)

    def start(self):
        if self.best_weights != '':
            # load best weights
            self.model.load_weights(self.best_weights)
        self.create_predictions()
        self.test_loss, self.test_acc = self.model.evaluate(self.x_test, self.y_test)

    def evaluate(self):
        """
        Run all evaluation methods
        """
        if self.history:
            # skip if no history is defined
            self.plot_acc()

            self.plot_loss()
        self._conf_matrix = self.conf_matrix(False)
        self.calculate_hter(False)

        self.predicted_labels_post_processed = self.smoothen_specs()
        self._conf_matrix_post_processed = self.conf_matrix(True)
        self.calculate_hter(True)
