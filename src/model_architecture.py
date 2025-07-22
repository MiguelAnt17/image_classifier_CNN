from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    """
    Build the CNN model

    Args:
        input_shape (tuple): Input format (ex: (224, 224, 3)).
        num_classes (int): Number of output classes.
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),

        # Convolutional Block 1: extract simple features
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 2: extract complex features
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 3: extract more complex features
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 4: more depth
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the convolution block to an 1D vector
        layers.Flatten(),

        # Dense layer to learn the features combinations
        # dropout to help against overfitting
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),

        # Output layer
        # Number of neurons = number of classes
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


if __name__ == '__main__':
    # Testing Values
    DUMMY_INPUT_SHAPE = (224, 224, 3)
    DUMMY_NUM_CLASSES = 10 
    
    print("Creating the model")
    model = create_model(DUMMY_INPUT_SHAPE, DUMMY_NUM_CLASSES)
    
    print("\nModel Summary:")
    model.summary()