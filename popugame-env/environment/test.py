from pettingzoo.test import api_test
from popugame_env import PopuGameEnv

def test_popugame_env():
    env = PopuGameEnv()
    api_test(env, num_cycles=100)
    env.close()

if __name__ == "__main__":
    test_popugame_env()
    print("PopuGame environment API test passed successfully.")