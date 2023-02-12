import multiprocessing as mp
from rich import print

class EnvWorker(mp.Process):
    def __init__(self, pipe, env_generator, worker_id, max_episode_length):
        super(EnvWorker, self).__init__()

        self.pipe = pipe
        self.env_generator = env_generator
        self.worker_id = worker_id
        self.max_episode_length = max_episode_length

        self.curr_goal = None
        self.step_count = 0

    def init_process(self):
        self._envs, self.env_names = self.env_generator()

    def run(self):
        self.init_process()

        while True:

            command, args = self._recv_message()

            if command == "eval_begin":
                if self._envs is None:
                    self.init_process()

                goal, env_name = args
                if type(env_name) == int:
                    env_name = self.env_names[env_name]
                self._env = self._envs[env_name]
                self.curr_goal = goal
                print(f"[eval] Worker #{self.worker_id} evaluating goal <{goal}> in env <{env_name}>")

                self.step_count = 0

                obs = self._env.reset()

                self._send_message("request_action", (goal, obs, self.step_count))

            elif command == "eval_step":
                action = args

                obs, reward, env_done, info = self._env.step(action)

                self.step_count += 1
                if self.step_count >= self.max_episode_length:
                    env_done = True

                self._send_message("request_action_with_info", 
                                   (goal, obs, reward, env_done, info, self.step_count))

            elif command == "close_env":
                self._env = None
                for env_name in self.env_names:
                    self._envs[env_name].close()
                self._envs = None
                self.env_names = None

            elif command == "kill_proc":
                return

    def _send_message(self, command, args):
        self.pipe.send((command, args))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, args = self.pipe.recv()

        return command, args