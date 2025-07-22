import tensorflow as tf

def load_datasets(base_dir, img_size, batch_size):
    """
    Upload the train, test and validation datasets

    Args:
        base_dir (str): directory for processed folder
        img_size (tuple): Images Lenght (ex: (224, 224)).
        batch_size (int)

    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names).
    """
    train_dir = f'{base_dir}/train'
    val_dir = f'{base_dir}/validation'
    test_dir = f'{base_dir}/test'

    print("Uploading datasets")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, label_mode='categorical', image_size=img_size, batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, label_mode='categorical', image_size=img_size, batch_size=batch_size, shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, label_mode='categorical', image_size=img_size, batch_size=batch_size, shuffle=False)
    
    class_names = train_ds.class_names
    print(f"Founded classes: {class_names}")

    # Normalization
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    def normalize_data(x, y):
        return normalization_layer(x), y

    normalized_train_ds = train_ds.map(normalize_data)
    normalized_val_ds = val_ds.map(normalize_data)
    normalized_test_ds = test_ds.map(normalize_data)

    # Otimizing Performance
    AUTOTUNE = tf.data.AUTOTUNE
    normalized_train_ds = normalized_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalized_val_ds = normalized_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalized_test_ds = normalized_test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    print("Datasets uploaded.")
    return normalized_train_ds, normalized_val_ds, normalized_test_ds, class_names