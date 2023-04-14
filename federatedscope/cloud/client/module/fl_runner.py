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
        self.conda = osp(config.conda.path, 'bin', 'conda')
        self.env = osp(self.config.conda.path, 'envs', config.conda.env_name)
        self.python = osp(self.env, 'bin', 'python')
        self.pip = osp(self.env, 'bin', 'pip')

    def _check_and_install(self, check_cmd, install_cmd, update_cmd=""):
        res = self.runner.run(check_cmd, warn=True)
        failed_groud = []
        for connection, result in res.items():
            if result.ok:
                self.logger.info(f"{connection.host} already has installed"
                                 f" ('{check_cmd}').")
            else:
                failed_groud.append(connection)
        if len(failed_groud):
            tg = ThreadingGroup.from_connections(failed_groud)
            tg.run(install_cmd)
            self.logger.info(f"Successfully installed with '{install_cmd}'.")

        if len(update_cmd):
            self.runner.run(update_cmd, warn=True)

    def _install_conda(self):
        check_cmd = f"{self.conda} --version"
        install_cmd = f"wget https://repo.anaconda.com/miniconda/Miniconda3" \
                      f"-py39_23.1.0-1-Linux-x86_64.sh; bash " \
                      f"Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -u -b -p" \
                      f" {self.config.conda.path}"
        self._check_and_install(check_cmd, install_cmd)

    def _install_python(self):
        check_cmd = f"{self.conda} env list | " \
                    f"grep {self.env}"

        install_cmd = f"{self.conda} " \
                      f"create -n {self.config.conda.env_name} " \
                      f"python={self.config.conda.python} -y"
        self._check_and_install(check_cmd, install_cmd)

    def _conda_install_package(self, package, extra_cmd=""):
        check_cmd = f"{self.conda} list " \
                    f"-n {self.config.conda.env_name} " \
                    f"| grep {package}"
        install_cmd = f"{self.conda} install -y {package} {extra_cmd}"

        self._check_and_install(check_cmd, install_cmd)

    def _pip_install_package(self, package, extra_cmd="", path=""):
        check_cmd = f"{self.conda} list " \
                    f"-n {self.config.conda.env_name} " \
                    f"| grep {package}"
        if len(path):
            install_cmd = f"cd {path}; " \
                          f"{self.pip} install {extra_cmd} ."
        else:
            install_cmd = f"{self.pip} install {extra_cmd} {package}"

        self._check_and_install(check_cmd, install_cmd)

    def _install_backend(self):
        if self.config.backend.type != 'torch':
            raise NotImplementedError(f"{self.config.backend.type}")

        check_cmd = f"{self.conda} list " \
                    f"-n {self.config.conda.env_name} pytorch | " \
                    f"grep {self.config.backend.pytorch}"
        install_cmd = f"{self.conda} install -y " \
                      f"-n {self.config.conda.env_name} " \
                      f"pytorch={self.config.backend.pytorch} " \
                      f"torchvision={self.config.backend.torchvision} " \
                      f"torchaudio={self.config.backend.torchaudio} " \
                      f"torchtext={self.config.backend.torchtext} " \
                      f"cudatoolkit={self.config.backend.cudatoolkit} " \
                      f"-c pytorch -c conda-forge"
        self._check_and_install(check_cmd, install_cmd)

    def _download_repo(self, repo_url, branch, path):
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        check_cmd = f"test -d {osp(path, repo_name, '.github')} || " \
                    f"test -d {osp(path, repo_name, 'README.md')}"
        install_cmd = f"rm -rf {osp(path, repo_name)}; " \
                      f"git clone {repo_url} {osp(path, repo_name)}"
        update_cmd = f"cd {osp(path, repo_name)}; " \
                     f"git checkout {branch}; " \
                     f"timeout 10 git fetch; " \
                     f"timeout 10 git pull"
        self._check_and_install(check_cmd, install_cmd, update_cmd)

    def _run_python(self, command):
        result = self.runner.run(f"{self.python} -c "
                                 f"'{command}'")
        self.logger.info(f"{self.runner.host} {result.stdout.strip()}")


if __name__ == '__main__':
    pass
