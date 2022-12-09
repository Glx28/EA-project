import numpy as np
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide tf information messages
import tensorflow as tf


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('target')
    df = {key: np.array(value)[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
        index = tf.keras.layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))


class DeepLearningModel:
    def __init__(self):
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.encoded_features = []
        self.all_inputs = []
        self.model = None
        self.n_epochs = 10
        self.activation_dct = {
            1: 'relu',
            2: 'sigmoid',
            3: 'softmax',
            4: 'softplus',
            5: 'softsign',
            6: 'tanh',
            7: 'selu',
            8: 'elu',
            9: 'exponential'
        }
        self.preprocessing()

    def preprocessing(self, debug_msg=False):
        tf.get_logger().setLevel('ERROR')

        dataframe = pd.read_csv('..\covid.csv')
        if debug_msg:
            print(dataframe.head())

        dataframe['target'] = np.where(dataframe['date_died'] == '9999-99-99', 0, 1)
        if debug_msg:
            print(dataframe['target'])
        dataframe = dataframe.drop(columns=['date_died', 'entry_date', 'id', 'date_symptoms'])

        train, val, test = np.split(dataframe.sample(frac=1), [int(0.8 * len(dataframe)), int(0.9 * len(dataframe))])

        if debug_msg:
            print(len(train), 'training examples')
            print(len(val), 'validation examples')
            print(len(test), 'test examples')

        batch_size = 256
        self.train_ds = df_to_dataset(train, batch_size=batch_size)
        self.val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
        self.test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

        [(train_features, label_batch)] = self.train_ds.take(1)
        if debug_msg:
            print('Every feature:', list(train_features.keys()))
            print('A batch of ages:', train_features['age'])
            print('A batch of targets:', label_batch)

        categorical_cols = ['age', 'sex', 'patient_type', 'intubed', 'pneumonia', 'pregnancy', 'diabetes', 'copd',
                            'asthma',
                            'inmsupr', 'hypertension',
                            'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco',
                            'contact_other_covid',
                            'covid_res', 'icu']
        for header in categorical_cols:
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
            encoding_layer = get_category_encoding_layer(name=header,
                                                         dataset=self.train_ds,
                                                         dtype='int64',
                                                         max_tokens=5)
            encoded_categorical_col = encoding_layer(categorical_col)
            self.all_inputs.append(categorical_col)
            self.encoded_features.append(encoded_categorical_col)

    def build(self, specification):
        if len(specification) != 17:
            raise Exception("Error. Specification must have 17 integer values.")

        # specification: [num_layers, num_neurons1, activation_func1, ..., num_neurons8, activation_func8]
        num_layers = specification[0]
        layers = tf.keras.layers.concatenate(self.encoded_features)  # input layer
        for i in range(num_layers):
            num_neurons = specification[2 * i + 1]
            activation_func = self.activation_dct[specification[2 * i + 2]]
            print(f"Layer {i + 1}: {num_neurons} neurons, {activation_func}")
            layers = tf.keras.layers.Dense(num_neurons, activation=activation_func)(layers)  # middle layers

        output = tf.keras.layers.Dense(1)(layers)  # output layer

        self.model = tf.keras.Model(self.all_inputs, output)
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=["accuracy"])

    def evaluate(self):
        self.model.fit(self.train_ds, epochs=self.n_epochs, validation_data=self.val_ds)
        loss, accuracy = self.model.evaluate(self.test_ds)

        return accuracy


def main():
    dlm = DeepLearningModel()

    spec = [4, 16, 1, 64, 5, 128, 8, 128, 1, 512, 1, 512, 2, 512, 1, 1024, 9]
    dlm.build(spec)
    acc = dlm.evaluate()
    print("Accuracy", acc)

    spec2 = [2, 1024, 1, 8, 5, 128, 8, 128, 1, 512, 1, 512, 2, 512, 1, 1024, 9]
    dlm.build(spec2)
    acc = dlm.evaluate()
    print("Accuracy", acc)


if __name__ == '__main__':
    main()
