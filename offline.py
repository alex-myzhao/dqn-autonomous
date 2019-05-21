from agent import Agent

if __name__ == '__main__':
    oa = Agent()
    oa.batch_size = 1000
    # oa.load('./model/offline_model.h5')
    oa.read_memory('./_out/memory.npy')
    oa.offline_learn(100)
    # oa.save('./model/offline_model.h5')
