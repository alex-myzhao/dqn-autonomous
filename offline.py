from agent import Agent

if __name__ == '__main__':
    oa = Agent()
    oa.read_memory('./_out/memory.npy')
    oa.offline_learn(200)
