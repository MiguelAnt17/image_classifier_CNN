import tensorflow as tf

train_dir = 'C:\\Users\\Programmer\\processed\\train'
val_dir = 'C:\\Users\\Programmer\\processed\\validation'
test_dir = 'C:\\Users\\Programmer\\processed\\test'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Upload the datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Verification
class_names_train = train_dataset.class_names
print("\n------------------------------------------------------------")
print("CLASSES VERIFICATION:")
print(f"Classes founded on the train dataset:\n{class_names_train}")
print(f"Number of classes founded: {len(class_names_train)}")

if len(class_names_train) > 1:
    print("\nClasses uploaded.")
else:
    print("\nClass not found - Verify the directory.")
print("------------------------------------------------------------\n")

class_names_test = test_dataset.class_names
print("\n------------------------------------------------------------")
print("CLASSES VERIFICATION:")
print(f"Classes founded on the test dataset:\n{class_names_test}")
print(f"Number of classes founded: {len(class_names_test)}")

if len(class_names_test) > 1:
    print("\nClasses uploaded.")
else:
    print("\nClass not found - Verify the directory.")
print("------------------------------------------------------------\n")

class_names_val = train_dataset.class_names
print("\n------------------------------------------------------------")
print("CLASSES VERIFICATION:")
print(f"Classes founded on the validation dataset:\n{class_names_val}")
print(f"Number of classes founded: {len(class_names_val)}")

if len(class_names_val) > 1:
    print("\nClasses uploaded.")
else:
    print("\nClass not found - Verify the directory.")
print("------------------------------------------------------------\n")

# Normalization layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_test_ds = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# For better performance (pre-upload the data)
AUTOTUNE = tf.data.AUTOTUNE
normalized_train_ds = normalized_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalized_val_ds = normalized_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalized_test_ds = normalized_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\nNormalization ended.")