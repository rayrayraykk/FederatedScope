from os.path import join as osp
from fabric import ThreadingGroup

from federatedscope.cloud.client.config import CONFIG
from federatedscope.cloud.client.module.logger import GRLogger


class FLRunnerManager(object):
    def __init__(self, hosts, user, password, port, logger, config):
        self.runner = ThreadingGroup(*hosts,
                                     user=user,
                                     port=port,
                                     connect_kwargs={"password": password})
        self.logger = logger
        self.config = config

    def _check_and_install(self, check_cmd, install_cmd):
        res = self.runner.run(check_cmd, warn=True)
        failed_groud = []
        for connection, result in res.items():
            print(check_cmd, result.ok)
            if result.ok:
                self.logger.info(f"{connection.host} already has installed"
                                 f" ('{check_cmd}').")
            else:
                failed_groud.append(connection)
        if len(failed_groud):
            tg = ThreadingGroup.from_connections(failed_groud)
            print(install_cmd)
            tg.run(install_cmd)
            self.logger.info(f"Successfully installed with '{install_cmd}'.")

    def _install_conda(self):
        check_cmd = f"{osp(self.config.conda.path, 'bin/conda')} --version"
        install_cmd = f"wget https://repo.anaconda.com/miniconda/Miniconda3" \
                      f"-py39_23.1.0-1-Linux-x86_64.sh; bash " \
                      f"Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -u -b -p" \
                      f" {self.config.conda.path}"
        self._check_and_install(check_cmd, install_cmd)

    def _install_python(self):
        env_name = self.config.conda.env_name
        check_cmd = f"{osp(self.config.conda.path, 'bin/conda')} env list | " \
                    f"grep {osp(self.config.conda.path, 'envs', env_name)}"

        install_cmd = f"{osp(self.config.conda.path, 'bin/conda')} " \
                      f"create -n {self.config.conda.env_name} " \
                      f"python={self.config.conda.python} -y"
        self._check_and_install(check_cmd, install_cmd)

    def _download_repo(self, repo):
        # TBD
        ...

    def _install_package(self, package):
        # conda activate
        check_cmd = f"{self.config.conda.path}/bin/conda list | grep {package}"
        install_cmd = f"{self.config.conda.path}/bin/conda " \
                      f"install {package} -y"
        self._check_and_install(check_cmd, install_cmd)

    def _run_python(self, command):
        result = self.runner.run(f"{self.config.conda.path}/bin/python -c "
                                 f"'{command}'")
        self.logger.info(f"{self.runner.host} {result.stdout.strip()}")


if __name__ == '__main__':
    a = FLRunnerManager(['47.93.123.201'], 'root', 'ay!8aD9g', 22, GRLogger(),
                        CONFIG)
    a._install_conda()
    a._install_python()
