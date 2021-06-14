import os
import threading
import numpy as np

class DummyGraph:
    def as_default(self): return self
    def __enter__(self): pass
    def __exit__(self, type,value, traceback): pass

def set_session(sess):pass

graph = DummyGraph()
sess = None


if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling, Flattern
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.backend import set_session
    import tensorflow as tf
    # 기본 그래프 객체

    graph = tf.get_default_graph()
    # Session 클래스 객체
    sess = tf.compat.v1.Session()

elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flattern
    from keras.optimizers import SGD


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr =0.001, shared_network=None, activation='sigmoid', loss='mse'):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None

    def predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flattern()


    def train_on_batch(self, x, y ):
        loss = 0.
        # with은 context manager의 _enter__함수를 호출하고 __exit__ 함수를 호출한다
        # 스레드 간 간섭 없이 작업 수행을 위함
        #with <lock class 객체>
        #    : thread-safe 객체
        with self.lock:
            with graph.as_default():
                set_session()
            loss = self.model.train_on_batch(x,y)
        return loss

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
    # classmethod와 일반 메서드 차이
    # https://kwonkyo.tistory.com/243#:~:text=%ED%81%B4%EB%9E%98%EC%8A%A4%20%EC%95%88%EC%97%90%20%EC%9E%88%EC%A7%80%EB%A7%8C%20%EC%9D%BC%EB%B0%98%20%ED%95%A8%EC%88%98,%EA%B8%B0%EB%8A%A5%EC%A0%95%EB%8F%84%EB%A1%9C%20%EC%82%AC%EC%9A%A9%EB%90%9C%EB%8B%A4%EA%B3%A0%20%ED%95%A9%EB%8B%88%EB%8B%A4.
    # 클래스메서드이므로 다른 형태의 파라미터도 전달하여 인스턴스 간 공유가 가능한 클래스 데이터 생성 가능
    @classmethod
    def get_shared_networks(cls, net='dnn', num_steps=1 , input_dims =0):
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            if net == 'dnn':
                # dnn  클래스가 어떠한 구조로 신경망 구축하는 지 보려면 get_network_head
                return DNN.get_network_head(Input((input_dim,)))
            elif net == 'lstm':
                return LSTMNetowrk.get_network_head(
                    Input((num_steps, input_dims))
                )
            elif net == 'cnn':
                return CNN.get_network_head(
                    Input((1, num_steps, input_dims))
                )

class DNN(Network):
    # *args 여러 개의 인자를 튜플 형태로 받는다
    # **kwargs 여러 개의 인자를 딕셔너리 형태로 받는다 . myname='seongmin' 이런 형태, myname이 key가 되고 seongimn은 values가 된다

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.input_dim,))
                output = self.get_shared_networks(inp).output

            else:
                inp = self.shared_network.input
                output = self.shared_network.output

            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal'(output)
            )
            self.model = Model(inp, output)
            self.model.compile(optimizer=SGD(lr=self.lr), loss=self.loss)

    # staticmethod는 일반 함수와 다를 게 없지만 인스턴스에서 메서드로 호출할 수 있다는 것과 해당 클래스와 연관성을 표현한다는 정도로 사용된다 .
    @staticmethod
    def get_network_head(inp):
        output = Dense(256, activation='sigmoid',
                       kernel_initializer='random_normal')(inp)
        # 배치 정규화 은닉 레이어 입력 정규화 하여 학습 가속화
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(128 , activation='sigmoid',
                        kernel_initializer='random_normal')
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid',
                    kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid',
                       kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)


    def train_on_batch(self, x, y ):
        # 행 수와 관계없이 열을 배열.reshape(변경할 배열 , 차원) 배열 생략 시 차원의 행 생략 시 남은 열로부터 추정
        # numpy array 함수는 N-dimensional array(ndarray)로 생성
        x = np.array(x).reshape((-1,self.input_dim))
        return super().train_on_batch(x,y )

    def predict(self,sample):
        sample = np.array(sample).reshape((1,self.input_dim))
        return super().predict(sample)

class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        #부모 클래스 생성자 호출할 때 사용
        super().__init__(*args ,**kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None

        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation,
            kernel_initializer='random_normal')(output)
        self.model = Model(inp,output)
        self.model.compile(
            optimizer=SGD(lr=self.lr, loss=self.loss)
        )

    @staticmethod
    def get_network_head(inp):
        output = LSTM(256, dropout=0.1,
                      return_sequences=True, stateful=False
                      ,kernel_initializer='random_normal')(inp)
        output =BatchNormalization()(output)
        output = LSTM(128, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = LSTM(64, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = LSTM(32, dropout=0.1,
                      return_sequences=True, stateful=False,
                      kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        return Model(inp,output)

    def train_on_batch(self, x, y ):
        # 3차원 변경
        x = np.array(x).reshape((-1,self.num_steps,self.input_dim))
        return super().train_on_batch(x,y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (1,self.num_steps, self.input_dim))
        return super().predict(sample)

class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim, 1))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    @staticmethod
    def get_network_head(inp):
        output = Conv2D(256, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(128, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(64, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(32, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Flattern()(output)
        return Model(inp, output)

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (-1, self.num_steps, self.input_dim, 1))
        return super().predict(sample)
