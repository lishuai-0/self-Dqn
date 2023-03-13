
import tensorflow as tf


def create_lstm_dqn_network(input_width: int, num_actions: int, lstm_width: int = 32, relu_width: int = 32):
    """
    Creates a lstm dqn network

    Args:
        input_width: Network input width
        num_actions: 行为数
        lstm_width: Number of LSTM layer units
        relu_width: Number of RELU layer units

    Returns: tensorflow keras Model

    """
    # 创建输入层，指定输入形状为（None，input_width），其中None表示输入序列的长度可以是任意值
    input_layer = tf.keras.layers.Input(shape=(None, input_width))

    # 创建LSTM层，指定输出宽度为lstm_width，将输入连接到此层
    lstm_layer = tf.keras.layers.LSTM(lstm_width)(input_layer)

    # 创建ReLU激活层，指定输出宽度为relu_width，将LSTM层的输出连接到此层
    relu_layer = tf.keras.layers.Dense(relu_width, activation='relu')(lstm_layer)

    # 创建Q层，指定输出宽度为num_actions，激活函数为线性，加入L1正则化，将ReLU层的输出连接到此层
    q_layer = tf.keras.layers.Dense(num_actions, activation='linear',
                                    kernel_regularizer=tf.keras.regularizers.l1())(relu_layer)

    # 创建模型，指定模型名称为'LSTM_Dqn'，将输入层和Q层连接到模型
    return tf.keras.Model(name='LSTM_Dqn', inputs=input_layer, outputs=q_layer)
