# 2. Repository Access and SSH

Some repositories are private, and the Docker builds download them over SSH. Before building:

1. Your GitHub account needs access to the private repositories (please contact tkimura4@illinois.edu or jinyang7@illinois.edu).
2. The edge server needs an SSH key that GitHub accepts, loaded in an **ssh-agent**.
3. The edge server needs password-free SSH to every edge device.

## Repositories

| Repository | Private? | Used for |
| --- | --- | --- |
| `acies-os/acies-os` | no | Main code, Docker files, `.env` templates. Cloned on the edge server and every device. |
| `acies-os/ui` | yes | Web UI. Cloned on the edge server only. |
| `acies-os/FoundationSense` | yes | Downloaded automatically while building the Docker images. |
| `acies-os/SPAR` | yes | Downloaded automatically while building the Docker images. |

**Getting access.** Send your GitHub username to an `acies-os` organization admin (tkimura4@illinois.edu or jinyang7@illinois.edu). Accept the
invitation, then confirm you can open all four repositories on GitHub.

## SSH key for GitHub (edge server)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github
cat ~/.ssh/id_ed25519_github.pub      # paste into GitHub → Settings → SSH and GPG keys
```

Add to `~/.ssh/config`:

```text
Host github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
```

Check it:

```bash
ssh -T git@github.com                                            # "Hi <username>! ..."
git ls-remote git@github.com:acies-os/FoundationSense.git HEAD   # prints a commit hash
```

## Load the key into ssh-agent (needed for every image build)

The image builds borrow your key from the ssh-agent instead of copying it into the image. If the
agent is not running or holds no key, the build fails while downloading dependencies.

```bash
eval "$(ssh-agent -s)"            # fish: eval (ssh-agent -c)
ssh-add ~/.ssh/id_ed25519_github
ssh-add -l                        # must list your key
```

## SSH from the edge server (server or laptop) to each device

The deploy commands log in to the devices by the names in `PI_HOSTS` (e.g. `rs1`). Give each
device an entry in the edge server's `~/.ssh/config`:

```text
Host rs1
    HostName 192.168.1.21        # the device's IP address
    User pi                      # the user you created on the device
    ForwardAgent yes             # lets the device clone the private repo using your key
Host rs2
    HostName 192.168.1.22
    User pi
    ForwardAgent yes
```

Copy your key to each device once (asks for the device password):

```bash
ssh-copy-id rs1
ssh rs1 hostname                 # must work without a password
ssh rs1 'ssh -T git@github.com'  # must greet you: the device can reach GitHub through your key
```

Devices never need their own GitHub key. Don't copy your private key onto them.
