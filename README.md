# Book-Recommendation

Dataset: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

## Contributor
* Kwak Seoyeon
* Kim Gyua
* Kim Dohun
* Kim Yiyoung

## Business Objective
> A recommendation system is a type of information filtering technology that recommends information that may be of interest to specific customers

## Data Exploration
### Books
![image](https://github.com/kdhhuns2000/book-recommendation/assets/52370750/58ebefab-9a4c-4e2b-a5cb-e641921c4665)
|Column|Description|
|--|--|
|ISBN|Book ISBN|
|Book-Title|Book Title|
|Book-Author|Book Author|
|Year-Of-Publication|year of publication|
|Publisher|publisher of the book|
|Image-URL-S|small image of the book , amazon link|
|Image-URL-M|medium size image of the book , amazon link|
|Image-URL-L|large image size of the book , amazon link|

### Ratings
![image](https://github.com/kdhhuns2000/book-recommendation/assets/52370750/de813b1b-ed14-42c3-a850-568e82cb2393)
|Column|Description|
|--|--|
|User-ID|Unique user id|
|ISBN|Book ISBN|
|Book-Rating|rating|

### Users
![image](https://github.com/kdhhuns2000/book-recommendation/assets/52370750/c496a218-9e41-463a-b8af-135e3c102081)
|Column|Description|
|--|--|
|User-ID|Unique user id|
|Location|location of the user|
|Age|User Age|

## Data Preprocessing
  - Delete unnecessary data and merge books dataset and ratings dataset to check which books users have rated
  - Recommended more than 250 times are selected
  - Cosine similarity measuring

## Data Modeling and evaluation
### Perceptron model
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
---
### Evaluation
![image](https://github.com/kdhhuns2000/book-recommendation/assets/52370750/c59eb0c8-47fe-4070-a114-53577b2149ab)
<br>

---
![image](https://github.com/kdhhuns2000/book-recommendation/assets/52370750/5d0b9a43-d45a-4a93-ba96-32d25250f9c2)
