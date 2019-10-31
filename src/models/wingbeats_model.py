from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GaussianNoise


def add_gaussian_noise(model, noise_std, input_shape):
    model.add(GaussianNoise(noise_std,  input_shape=(input_shape, 1)))


def WingbeatsNetModel(plot=True, num_classes=2,
                      input_shape=int(0.25*44100), blocks=1,
                      batch_norm=True,
                      dropout=True, use_noise=True, noise_std=320,
                      padding="valid", opt=None, lr=None,
                      filter_size=3):
    """A 1D '5 layer CNN' conv net, as used in the paper
        'Mosquito wingbeat analysis and classification usingdeep learning'

    The definition of the net were taken from
        https://www.kaggle.com/potamitis/deep-1-0-91-acc

    If a Optimizer (opt) is passed, we use it insteaf of the Adam default;

    If Learning Rate (lr) is None, we use the default lr, otherwise we use
        the passed value
    """

    # Helper function to check for adding batch norm
    def add_batch_norm(model):
        if batch_norm:
            model.add(BatchNormalization())

    # Helper function for adding convs with less lines
    def add_conv(model, filters, first=False):
        if first:
            model.add(Conv1D(filters, filter_size, activation='relu',
                             kernel_initializer="he_normal",
                             input_shape=(input_shape,1) ))
        else:
            model.add(Conv1D(filters, filter_size, activation='relu',
                             kernel_initializer="he_normal"))

    if not isinstance(blocks, int):
        raise ValueError(f"blocks must be int, found {type(blocks)}")
    valid_blocks = [1, 2, 3, 4, 5]
    if blocks not in valid_blocks:
        raise ValueError(f"blocks must one of {valid_blocks}")

    model = Sequential()

    # Add either noise or first conv block
    if use_noise:
        add_gaussian_noise(model, noise_std, input_shape)
    add_conv(model, 16, first=not use_noise)
    add_conv(model, 16)
    add_batch_norm(model)


    # Keep adding conv blocks
    if blocks > 1:
        add_conv(model, 32)
        add_conv(model, 32)
        add_batch_norm(model)
        model.add(MaxPooling1D(2))

    if blocks > 2:
        add_conv(model, 64)
        add_conv(model, 64)
        add_batch_norm(model)
        model.add(MaxPooling1D(2))

    if blocks > 3:
        add_conv(model, 128)
        add_conv(model, 128)
        add_batch_norm(model)
        model.add(MaxPooling1D(2))

    if blocks > 4:
        add_conv(model, 256)
        add_conv(model, 256)
        add_batch_norm(model)

    # Agg pooling/dropout
    model.add(GlobalAveragePooling1D())
    if dropout:
        model.add(Dropout(0.5))

    # If 2 classes suppose independent and use sigmoid with binarycrossentropy,
    #   otherwise use softmax
    if num_classes == 2:
        last_layer = Dense(1, activation='sigmoid')
        loss = "binary_crossentropy"
    else:
        last_layer = Dense(num_classes, activation='softmax')
        loss = "categorical_crossentropy"
    model.add(last_layer)

    if opt is None:
        if lr is None:
            opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        else:
            opt = Adam(learning_rate=lr)

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    # Plot model
    if plot:
        plot_model(model, to_file='wingbeats_model.png')

    return model


if __name__ == "__main__":
    WingbeatsNetModel()
