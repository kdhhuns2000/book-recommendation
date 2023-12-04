import argparse
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import numpy as np
import sagemaker

class RecommenderNet(tf.keras.Model):
    # function initialization
    def __init__(self, num_users, num_book_title, embedding_size, dropout_rate=0.2, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_book_title = num_book_title
        self. embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.user_embedding = layers.Embedding( # user embedding layer
            num_users,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer =keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
        self.book_title_embedding = layers.Embedding( # book_title embedding layer
            num_book_title,
            embedding_size,
            embeddings_initializer = 'he_normal',
            embeddings_regularizer =keras.regularizers.l2(1e-6)
        )
        self.book_title_bias = layers.Embedding(num_book_title, 1) # layer embedding book_title
        self.dropout = layers.Dropout(rate=dropout_rate)
        
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0]) # call embedding layer 1
        user_vector = self.dropout(user_vector)
        user_bias = self.user_bias(inputs[:, 0]) # call embedding layer 2
        book_title_vector = self.book_title_embedding(inputs[:, 1]) # call embedding layer 3
        book_title_vector = self.dropout(book_title_vector)
        book_title_bias = self.book_title_bias(inputs[:, 1]) # call embedding layer 4
        dot_user_book_title = tf.tensordot(user_vector, book_title_vector, 2) # dot product multiplication
        x = dot_user_book_title + user_bias + book_title_bias
        return tf.nn.sigmoid(x) # activate sigmoid
    
 

    def train():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_dir', type=str)
        parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
        parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
        parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
        
        parser.add_argument('--xtrain', type=str, default='./x_train.npy')
        parser.add_argument('--xval', type=str, default='./x_val.npy')
        parser.add_argument('--ytrain', type=str, default='./y_train.npy')
        parser.add_argument('--yval', type=str, default='./y_val.npy')
        
        
        parser.add_argument('--epochs', type=int, default=3)
        parser.add_argument('--num_users', type=int, default=100)
        parser.add_argument('--num_book_title', type=int, default=100)
        parser.add_argument('--embedding_size', type=int, default=50)

        args, _ = parser.parse_known_args()


        # 데이터 로드
        x_train = np.load(os.path.join(args.train, 'X_train.npy'))
        x_val = np.load(os.path.join(args.train, 'X_val.npy'))
        y_train = np.load(os.path.join(args.train, 'y_train.npy'))
        y_val = np.load(os.path.join(args.train, 'y_val.npy'))

        # 모델 초기화
        model = RecommenderNet(args.num_users, args.num_book_title, args.embedding_size)

        # 컴파일 등 학습 설정
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        # 학습
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=50,
            epochs=10,
            validation_data=(x_val, y_val)
        )

        # 학습이 완료되면 모델을 저장
        model.save('/opt/ml/model')  # SageMaker에서 모델을 저장할 경로

    if __name__ == '__main__':
        train()