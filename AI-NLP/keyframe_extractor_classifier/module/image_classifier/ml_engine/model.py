import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..utils import custom_logging


class FrameClassifier(nn.Module):
    def __init__(self, framework="pytorch"):
        super(FrameClassifier, self).__init__()
        self.framework = framework
        self.logger = custom_logging.get_logger()
        self.initialize_network_layers()

    @staticmethod
    def get_output_shape(model, image_dim):
        return model(torch.rand(*image_dim)).data.shape

    @staticmethod
    def make_variable_names():
        num_convolution_layers = config.NETWORK["CNN"]["num_convolution_layers"]
        num_dense_layers = config.NETWORK["CNN"]["num_dense_layers"]
        variables = []
        for layer in range(max(num_convolution_layers, num_dense_layers)):
            if layer < num_convolution_layers and layer < num_dense_layers:
                variables.append((f"conv{layer + 1}", f"linear{layer + 1}"))
            elif num_convolution_layers > layer >= num_dense_layers:
                variables.append((f"conv{layer + 1}", ""))
            elif num_convolution_layers <= layer < num_dense_layers:
                variables.append(("", f"linear{layer + 1}"))
        return variables

    def get_first_linear_layer_input_features(self):
        conv1_output_shape = self.get_output_shape(
            self.max_pool,
            self.get_output_shape(self.conv1, config.INPUT_SHAPE),
        )
        conv2_output_shape = self.get_output_shape(
            self.max_pool,
            self.get_output_shape(self.conv2, conv1_output_shape),
        )
        conv3_output_shape = self.get_output_shape(
            self.max_pool,
            self.get_output_shape(self.conv3, conv2_output_shape),
        )
        conv4_output_shape = self.get_output_shape(
            self.max_pool,
            self.get_output_shape(self.conv4, conv3_output_shape),
        )
        return list(conv4_output_shape)

    def initialize_network_layers(self):
        if self.framework == "pytorch":
            variables = self.make_variable_names()
            setattr(self, "activation_function", nn.ReLU())
            setattr(self, "max_pool", nn.MaxPool2d(kernel_size=2))
            for i in range(len(variables)):
                if variables[i][0] != "":
                    setattr(
                        self,
                        variables[i][0],
                        nn.Conv2d(
                            in_channels=config.NETWORK["CNN"]["in_channels"][i],
                            out_channels=config.NETWORK["CNN"]["out_channels"][i],
                            kernel_size=config.NETWORK["CNN"]["kernel_sizes"][i],
                            stride=config.NETWORK["CNN"]["stride"][i],
                            padding=config.NETWORK["CNN"]["padding"][i],
                        ),
                    )
            for i in range(len(variables)):
                if variables[i][1] != "":
                    dense_input_output_features = config.NETWORK["CNN"][
                        "dense_layer_input_output_features"
                    ][i]
                    if dense_input_output_features[0] is None:
                        input_features = int(
                            np.prod(self.get_first_linear_layer_input_features())
                        )
                        setattr(
                            self,
                            variables[i][1],
                            nn.Linear(input_features, dense_input_output_features[1]),
                        )
                    else:
                        setattr(
                            self,
                            variables[i][1],
                            nn.Linear(
                                dense_input_output_features[0],
                                dense_input_output_features[1],
                            ),
                        )
        elif self.framework in ["tensorflow", "keras"]:
            model = Sequential()
            # Convolution and Padding layers
            model.add(layers.BatchNormalization(input_shape=(128, 128, 3)))
            model.add(
                layers.Convolution2D(
                    filters=32,
                    kernel_size=3,
                    activation="relu",
                    input_shape=(128, 128, 3),
                )
            )
            model.add(layers.MaxPooling2D(pool_size=2))
            model.add(
                layers.Convolution2D(
                    filters=64,
                    kernel_size=4,
                    padding="same",
                    activation="relu",
                )
            )
            model.add(layers.MaxPooling2D(pool_size=2))
            model.add(
                layers.Convolution2D(
                    filters=128,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                )
            )
            model.add(layers.MaxPooling2D(pool_size=2))
            model.add(
                layers.Convolution2D(
                    filters=128,
                    kernel_size=2,
                    padding="same",
                    activation="relu",
                )
            )
            model.add(layers.MaxPooling2D(pool_size=2))
            model.add(layers.Flatten())
            # Fully connected layers
            model.add(layers.Dense(units=128, activation="relu"))
            model.add(layers.Dense(units=64, activation="relu"))
            model.add(layers.Dense(units=32, activation="relu"))
            model.add(
                layers.Dense(
                    units=len(config.CLASSIFICATION_LABELS),
                    activation="softmax",
                )
            )
            setattr(self, "model", model)

        else:
            self.logger.exception(
                "only tensorflow, keras and pytorch frameworks are allowed"
            )

    def forward(self, inputs):
        if self.framework == "pytorch":
            conv1_output = self.max_pool(
                self.activation_function(self.conv1(inputs))
            )  # -->3*128*128 32*63*63
            conv2_output = self.max_pool(
                self.activation_function(self.conv2(conv1_output))
            )  # -->32*63*63 64*31*31
            conv3_output = self.max_pool(
                self.activation_function(self.conv3(conv2_output))
            )  # -->64*31*31 128*15*15
            conv4_output = self.max_pool(
                self.activation_function(self.conv4(conv3_output))
            )  # -->128*15*15 128*7*7
            flattened_output = conv4_output.view(conv4_output.size(0), -1)
            final_output = self.linear4(
                self.linear3(self.linear2(self.linear1(flattened_output)))
            )
            return final_output

    @classmethod
    def from_keras(cls):
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential

        model = Sequential()
        # Convolution and Padding layers
        model.add(layers.BatchNormalization(input_shape=(128, 128, 3)))
        model.add(
            layers.Convolution2D(
                filters=32, kernel_size=3, activation="relu", input_shape=(128, 128, 3)
            )
        )
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(
            layers.Convolution2D(
                filters=64, kernel_size=4, padding="same", activation="relu"
            )
        )
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(
            layers.Convolution2D(
                filters=128, kernel_size=3, padding="same", activation="relu"
            )
        )
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(
            layers.Convolution2D(
                filters=128, kernel_size=2, padding="same", activation="relu"
            )
        )
        model.add(layers.MaxPooling2D(pool_size=2))
        model.add(layers.Flatten())
        # Fully connected layers
        model.add(layers.Dense(units=128, activation="relu"))
        model.add(layers.Dense(units=64, activation="relu"))
        model.add(layers.Dense(units=32, activation="relu"))
        model.add(
            layers.Dense(units=len(config.CLASSIFICATION_LABELS), activation="softmax")
        )
        return cls(model=model)
