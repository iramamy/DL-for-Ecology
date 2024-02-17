class Evaluate:
    """
    A class for evaluating classification results using confusion matrices, ROC curves, and training/validation history plots.
    """

    def __init__(self, y_value, y_pred, history=None):
        """
        Args:
            y_value (numpy ndarray): True target labels.
            y_pred (numpy ndarra): Predicted target labels.
            history (History, optional): History object from the model.
        """
        self.y_value = y_value
        self.y_pred = y_pred
        self.history = history

    def create_confusion_matrix(self, ax=None, title=''):
        """
        Create and display the confusion matrix for the classification results.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot the confusion matrix on.
            title (string): Title for the confusion matrix plot.
        """
        conf_matrix = metrics.confusion_matrix(self.y_value, self.y_pred)
        display_conf = metrics.ConfusionMatrixDisplay(
            conf_matrix,
            display_labels=['call', 'no-call']
        )
        display_conf.plot(cmap='Blues', values_format='d', ax=ax)
        ax.set_title(title)

    def create_roc_curve(self, ax=None, title=''):
        """
        Create and display the ROC curve for the classification results.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot the ROC curve on.
            title (string): Title for the ROC curve plot.
        """
        FPR, TPR, thresholds = metrics.roc_curve(self.y_value, self.y_pred)
        roc_auc = metrics.auc(FPR, TPR)
        display_roc = metrics.RocCurveDisplay(fpr=FPR, tpr=TPR, roc_auc=roc_auc)
        display_roc.plot(ax=ax)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    def plot_history(self, ax=None, title=''):
        """
        Plot the training and validation history of the model.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot the history on.
            title (str): Title for the history plot.
        """
        if self.history is None:
            raise ValueError("History object is missing.")

        train_loss = self.history.history['loss']
        train_accuracy = self.history.history['accuracy']
        epochs = range(1, len(train_loss) + 1)

        color = 'tab:red'
        ax.plot(epochs, train_loss, color=color, label='Training')
        ax.set_xlabel('Epochs', fontsize=16, labelpad=5)
        ax.set_ylabel('Loss', fontsize=12, labelpad=5)
        ax.tick_params(color=color, length=4)
        ax.tick_params(axis='y', labelcolor=color)

        ax2 = ax.twinx()
        color = 'tab:blue'
        ax2.plot(epochs, train_accuracy, color=color, label='Training')
        ax2.tick_params(color=color, length=4)
        ax2.tick_params(axis='y', labelcolor=color)

        ax.set_title(title, fontsize=18, pad=20)
        ax.legend(loc='upper right', fontsize='large')
        ax2.legend(loc='lower left', fontsize='large')

def plot_evaluation_models(histories, evaluations, subplot_width=5, subplot_height=4):
    """
    Plot the training history, ROC curve, and confusion matrix for each model.

    Args:
        histories (list): List of History objects from the models.
        evaluations (list): List of Evaluate objects for the models.
        subplot_width (integer): Width of each subplot.
        subplot_height (integer): Height of each subplot.
    """

    num_models = len(histories)
    fig, axes = plt.subplots(3, num_models, figsize=(num_models*subplot_width, 3*subplot_height),
                             gridspec_kw={'width_ratios': [1]*num_models})

    for i in range(num_models):
        # Plot training history
        if i == 0:
            evaluations[i].plot_history(ax=axes[0, i])
            axes[0, i].set_ylabel("HISTORY \n \n \n Loss", fontsize=16, labelpad=20)  
        elif i == 3:
            evaluations[i].plot_history(ax=axes[0, i])
            axes[0, i].set_ylabel('Accuracy', fontsize=16, labelpad=-320)
        else:
            evaluations[i].plot_history(ax=axes[0, i])
            axes[0, i].set_ylabel('')  

        # Plot ROC curve
        evaluations[i].create_roc_curve(ax=axes[1, i])
        if i != 0:
            axes[1, i].set_ylabel('', fontsize=16)  
        else:
            axes[1, i].set_ylabel('ROC CURVE\n\n\n\n True Positive Rate', fontsize=16, labelpad=20)

        axes[1, i].set_xlabel('False Positive Rate', fontsize=16, labelpad=10)  

        # Plot confusion matrix
        evaluations[i].create_confusion_matrix(ax=axes[2, i])
        if i != 0:
            axes[2, i].set_ylabel('', fontsize=16)
        else:
            axes[2, i].set_ylabel('CONFUSION MATRIX \n\n\n\n True Label', fontsize=16, labelpad=10)

        axes[2, i].set_xlabel('Predicted Label', fontsize=16, labelpad=10)  

        # Set title for each subplot
        axes[0, i].set_title(f"Model {i+1}", fontsize=18, pad=20)

    plt.tight_layout(pad=4.0)
    plt.savefig('Figure.svg', format='svg')
    plt.show()

# How to run it    
# histories = [history_1, history_2, history_3, history_4]
# evaluation_1 = Evaluate(Y_test[:,1], Y_hat_1, history_1)
# evaluation_2 = Evaluate(Y_test[:,1], Y_hat_2, history_2)
# evaluation_3 = Evaluate(Y_test[:,1], Y_hat_3, history_3)
# evaluation_4 = Evaluate(Y_test[:,1], Y_hat_4, history_4ss)
# evaluations = [evaluation_1, evaluation_2, evaluation_3, evaluation_4]
# plot_evaluation_models(histories, evaluations)
