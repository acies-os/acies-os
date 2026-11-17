# Repository Access and SSH

Several Acies-OS repositories and Python dependencies are **private**. Builds pull them over SSH,
so you need three things before anything else works:

1. A GitHub account that belongs to the `acies-os` organization.
2. An SSH key on the edge server that GitHub accepts.
3. An **SSH agent** holding that key while you build images. The Docker builds forward the agent
   instead of copying keys into images.

You also need passwordless SSH from the edge server to every Pi so the deploy recipes can run
commands on them.

## 1. Repositories you need

| Repository | Visibility | Needed by | How it's fetched |
| --- | --- | --- | --- |
| `acies-os/acies-os` | private | edge server and every Pi | `git clone git@github.com:acies-os/acies-os.git` |
| `acies-os/ui` | private | edge server (builds `acies-ui` image) | `git clone git@github.com:acies-os/ui.git` |
| `acies-os/FoundationSense` | private | image builds (VibroFM classifiers) | `uv` via `git+ssh` during `docker build` |
| `acies-os/SPAR` (branch `demo2`) | private | image builds (SPAR classifier) | `uv` via `git+ssh` during `docker build` |
| `jinyangustc/rawshake` (tag `v0.1.15`) | public | image builds (geophone driver) | `uv` via `https` |

If any of the private repositories return `Repository not found` or `Permission denied
(publickey)`, the cause is almost always access or key setup, not the repository itself.

## 2. Get access to the organization

1. Send your GitHub username to an `acies-os` organization admin and ask to be added.
2. Accept the invitation at <https://github.com/acies-os> (or from the email GitHub sends).
3. Make sure your team grants **read** access to all four private repositories above. Org
   membership alone may not include `FoundationSense` and `SPAR`.

Check what you can see:

```bash
gh repo list acies-os --limit 50   # needs the GitHub CLI, or just open the org page
```

## 3. Create and register an SSH key (edge server)

Skip this step if `ssh -T git@github.com` already greets you by username.

```bash
ssh-keygen -t ed25519 -C "you@example.com" -f ~/.ssh/id_ed25519_github
cat ~/.ssh/id_ed25519_github.pub
```

Add the public key at **GitHub → Settings → SSH and GPG keys → New SSH key**.
If your organization enforces SAML SSO, click **Configure SSO → Authorize** next to the key.

Tell SSH to use it for GitHub (`~/.ssh/config`):

```text
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    IdentitiesOnly yes
    AddKeysToAgent yes
    # macOS only: remember passphrase in Keychain
    UseKeychain yes
```

Test it:

```bash
ssh -T git@github.com
# Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.

git ls-remote git@github.com:acies-os/FoundationSense.git HEAD   # must print a commit hash
git ls-remote git@github.com:acies-os/SPAR.git demo2              # must print a commit hash
```

## 4. Load the key into an SSH agent (required for Docker builds)

`just build-server` and `just build-pi` run `docker buildx build --ssh default`. That forwards
the agent named by `$SSH_AUTH_SOCK` into the build so `uv` can fetch the private dependencies.
**If no agent is running, or the agent has no key, the build fails at the `uv sync` step.**

```bash
# bash / zsh
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github

# fish
eval (ssh-agent -c)
ssh-add ~/.ssh/id_ed25519_github

ssh-add -l          # must list your key
```

`just start-ssh-agent` prints these commands as a reminder.

:::{tip}
On macOS the system agent is usually already running; `ssh-add --apple-use-keychain
~/.ssh/id_ed25519_github` adds the key once and keeps it across reboots.
:::

(sec-ssh-to-pi)=

## 5. SSH from the edge server to each Pi

The deploy recipes (`just deploy-pi`, `just registry-trust`, `just upload-model`,
`scripts/update-pis.sh`) use `ssh <host>` with **no password prompt**, and they use the names in
`PI_HOSTS` (for example `rs1`) directly as SSH targets. Give each Pi an alias in
`~/.ssh/config` on the edge server:

```text
Host rs1 rs2 rs3 rs5 rs6 rs7 rs8 rs10
    HostName %h.local          # mDNS name set in Raspberry Pi Imager; or use a fixed IP per host
    User myshake               # the username you set when flashing the Pi
    IdentityFile ~/.ssh/id_ed25519_pi
    ForwardAgent yes           # lets the Pi `git pull` the private repo with your GitHub key
    StrictHostKeyChecking accept-new
    ServerAliveInterval 15
```

If `.local` names are unreliable on your network, write one block per Pi with
`HostName 192.168.x.y` instead. See {ref}`Static addresses <static-addresses>`.

Install your key on each Pi (asks for the Pi password once):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_pi -N ""
for h in rs1 rs2 rs3; do ssh-copy-id -i ~/.ssh/id_ed25519_pi.pub "$h"; done

# verify: must print the hostname with no password prompt
for h in rs1 rs2 rs3; do ssh "$h" hostname; done
```

You can also paste the public key into Raspberry Pi Imager when flashing (Services → *Allow
public-key authentication only*). See [Pi setup](04-pi.md).

(sec-pi-github-access)=

## 6. GitHub access from the Pis

Each Pi needs a checkout of `acies-os` for the compose files and justfiles (`docker/*.yml`,
`just/*.just`). `scripts/update-pis.sh` also runs `git fetch` / `git pull` on the Pis. Pick
one way to give the Pis GitHub access:

* **Agent forwarding (recommended).** Set `ForwardAgent yes` as shown above. The Pi uses
  your laptop's key only while you're connected; no secret is stored on the Pi. Check with
  `ssh rs1 'ssh -T git@github.com'`.
* **Read-only deploy key per Pi.** Generate a key on the Pi and add it under
  *acies-os/acies-os → Settings → Deploy keys* (read access only). Use this if the Pis must
  pull while you're not connected.

Never copy your personal GitHub private key onto a Pi.

:::{note}
The Pis **do not** need access to `FoundationSense` or `SPAR`. Those are baked into the
`acies-pi` image when you build it on the edge server, and the Pis only pull the finished image.
:::

## Checklist

- [ ] `ssh -T git@github.com` succeeds on the edge server.
- [ ] `git ls-remote` works for `acies-os`, `ui`, `FoundationSense`, and `SPAR`.
- [ ] `ssh-add -l` lists your GitHub key in the shell where you'll run `just build-*`.
- [ ] `ssh rs1 hostname` (and every other Pi) works with no password.
- [ ] `ssh rs1 'ssh -T git@github.com'` succeeds (or a deploy key is installed).
