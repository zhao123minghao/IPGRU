import tensorflow as tf
from tensorflow.keras.layers import Dense,GRU,Input,Dropout,Lambda,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
def metrics_vae(y_true,y_pred):
    print('y_true: ', y_true, '\n', 'y_pred: ', y_pred)
    # print('y_true:',y_true.shape,'y_pred:',y_pred)
    omega = tf.sign(y_true)  # 0 if 0, 1 if > 0; not possibly < 0 in our data
    mse_nz = tf.reduce_mean(tf.multiply(tf.pow(y_true - y_pred, 2), omega))
    mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    return mse_nz

def sampling(args):
    z_mean,z_log_var=args
    epsilon=K.random_normal(shape=K.shape(z_mean))
    return z_mean+K.exp(z_log_var/2)*epsilon

def vae2(inputdims):
    input=Input(shape=(int(inputdims.shape[1]),))             #None是送入的样本数维度
    #outputs=input

    #VAE
    # outputs = [Dense(256, activation='relu')(output) for output in outputs]
    # outputs = [Dropout(0.5)(output) for output in outputs]
    # outputs = [Dense(128, activation='relu')(output) for output in outputs]
    # outputs = [Dropout(0.5)(output) for output in outputs]
    # outputs = [Dense(20, activation='relu')(output) for output in outputs]
    # # outputs = [Dense(20, activation='relu')(output) for output in outputs]
    # # outputs = [Lambda(sampling,output_shape=(20,))(output) for output in outputs]
    # # outputs = [Dropout(0.5)(output) for output in outputs]
    # outputs = [Dense(128, activation='relu')(output) for output in outputs]
    # outputs = [Dropout(0.5)(output) for output in outputs]
    # outputs = [Dense(256, activation='relu')(output) for output in outputs]
    # outputs = [Dropout(0.5)(output) for output in outputs]
    # outputs = [Dense(100,activation='softplus')(output) for output in outputs]

    # #Attention
    # query_input = tf.keras.Input(shape=(100,), dtype='int32')
    # value_input = tf.keras.Input(shape=(100,), dtype='int32')
    # # Embedding lookup.
    # token_embedding = tf.keras.layers.Embedding(input_dim=100, output_dim=64)
    # # Query embeddings of shape [batch_size, Tq, dimension].
    # query_embeddings = token_embedding(query_input)
    # # Value embeddings of shape [batch_size, Tv, dimension].
    # value_embeddings = token_embedding(value_input)
    # query_seq_encoding = Dense(100,activation='relu')(query_embeddings)
    # # Value encoding of shape [batch_size, Tv, filters].
    # value_seq_encoding = Dense(100,activation='relu')(value_embeddings)
    # query_value_attention_seq = tf.keras.layers.Attention()(
    #     [query_seq_encoding, value_seq_encoding])
    # # query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    # #     query_seq_encoding)
    # # query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    # #     query_value_attention_seq)
    # outputs=Dense(inputdims)(query_value_attention_seq)

    #deepimpute
    # outputs=Dense(512,activation='relu')(outputs)
    # outputs=Dense(256,activation='relu')(outputs)
    # outputs=Dropout(0.5)(outputs)
    # outputs=Dense(300,activation='softplus')(outputs)

    #Autoimpute
    # outputs=Dense(2000,activation='sigmoid')(input)
    # outputs=Dense(inputdims.shape[1])(outputs)

    # #DCA
    # outputs=Dropout(0.5)(input)
    # outputs=Dense(64,activation='relu')(outputs)
    # outputs=Dense(32,activation='relu')(outputs)
    # outputs=Dense(64,activation='relu',kernel_initializer='glorot_uniform')(outputs)
    # outputs=Dense(inputdims.shape[1],activation='relu')(outputs)


    #GRU
    timesteps = int(inputdims.shape[1])  # 这将是输入的时间步数
    features = int(inputdims.shape[2])    # 这将是特征的数量
    inputgru = Input(shape=(timesteps, features))


    outputs = GRU(768, return_sequences=True)(inputgru)    #原记忆体512
    outputs = Dropout(0.5)(outputs)
    outputs = GRU(640, return_sequences=False)(outputs)   #原记忆体256
    outputs = Dropout(0.5)(outputs)
    # outputs=Dense(200,activation='relu')(outputs)
    outputs = Dense(300, activation='softplus')(outputs)

    outputs=Flatten()(outputs)
    model = Model(inputs=inputgru, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=metrics_vae)
    return model