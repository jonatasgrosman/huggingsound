# Contributing to HuggingSound

We would love for you to contribute to HuggingSound and help make it even better than it is
today! As a contributor, here are the guidelines we would like you to follow:

 - [Code of Conduct](#coc)
 - [Found a Bug?](#issue)
 - [Missing a Feature?](#feature)
 - [Submission Guidelines](#submit)
 - [Coding Rules](#rules)
 - [Commit Message Guidelines](#commit)
 - [Building and Testing](#dev)

## <a name="coc"></a> Code of Conduct
Help us keep the project open and inclusive. Please read and follow our Code of Conduct.

As contributors and maintainers of the project, we pledge to respect everyone who contributes by posting issues, updating documentation, submitting pull requests, providing feedback in comments, and any other activities.

The contributors and maintainers communication must be constructive and never resort to personal attacks, trolling, public or private harassment, insults, or other unprofessional conduct.

We promise to extend courtesy and respect to everyone involved in this project regardless of gender, gender identity, sexual orientation, disability, age, race, ethnicity, religion, or level of experience. We expect anyone contributing to the project to do the same.

If any member of the community violates this code of conduct, the maintainers of the project may take action, removing issues, comments, and pull requests or blocking accounts as deemed appropriate.

If you are subject to or witness unacceptable behavior, or have any other concerns, please send a message to the maintainers of the project.

## <a name="issue"></a> Found a Bug?
If you find a bug in the source code, you can help us by
[submitting an issue](#submit-issue) to our [GitHub Repository](https://github.com/jonatasgrosman/huggingsound). Even better, you can
[submit a Pull Request](#submit-pr) with a fix.

## <a name="feature"></a> Missing a Feature?
You can *request* a new feature by [submitting an issue](#submit-issue) to our GitHub
Repository. If you would like to *implement* a new feature, please submit an issue with
a proposal for your work first, to be sure that we can use it.
Please consider what kind of change it is:

* For a **Major Feature**, first open an issue and outline your proposal so that it can be
discussed. This will also allow us to better coordinate our efforts, prevent duplication of work,
and help you to craft the change so that it is successfully accepted into the project.
* **Small Features** can be crafted and directly [submitted as a Pull Request](#submit-pr).

## <a name="submit"></a> Submission Guidelines

In our development process we follow the [GitHub flow][github-flow], that is very powerful and easy to understand. 
That process enforces continuous delivery by **making anything in the master branch deployable**.
So everybody needs to keep the master branch as safe as possible and ready to be deployed at any time.

### <a name="submit-issue"></a> Submitting an Issue

Before you submit an issue, please search the issue tracker, maybe an issue for your problem already exists and the discussion might inform you of workarounds readily available.

We want to fix all the issues as soon as possible, but before fixing a bug we need to reproduce and confirm it. In order to reproduce bugs, we will systematically ask you to provide a minimal reproduction scenario. In this scenario you need to describe how can we reproduce the bug and provide all the additional information that you think will help us to reproduce it.

A minimal reproduce scenario allows us to quickly confirm a bug (or point out coding problem) as well as confirm that we are fixing the right problem. And when is possible, please create a standalone git repository demonstrating the problem.

We will be insisting on a minimal reproduce scenario in order to save maintainers time and ultimately be able to fix more bugs. Interestingly, from our experience users often find coding problems themselves while preparing a minimal reproduction scenario. We understand that sometimes it might be hard to extract essentials bits of code from a larger code-base but we really need to isolate the problem before we can fix it.

Unfortunately, we are not able to investigate/fix bugs without a minimal reproduction, so if we don't hear back from you we are going to close an issue that doesn't have enough info to be reproduced.

You can file new issues by filling out our [new issue form](https://github.com/jonatasgrosman/huggingsound/issues/new).


### <a name="submit-pr"></a> Submitting a Pull Request (PR)
Before you submit your Pull Request (PR) consider the following guidelines:

1. Search [GitHub](https://github.com/jonatasgrosman/huggingsound/pulls) for an open or closed PR
  that relates to your submission. You don't want to duplicate effort.
1. Fork the jonatasgrosman/huggingsound repo.
1. Make your changes in a new git branch:

     ```shell
     git checkout -b my-fix-branch master
     ```

1. Create your patch, **including appropriate test cases**.
1. Follow our [Coding Rules](#rules).
1. Run the full test suite, as described in the section [Building and Testing](#dev),
  and ensure that all tests pass.
1. Commit your changes using a descriptive commit message that follows our
  [commit message conventions](#commit). Adherence to these conventions
  is necessary because release notes are automatically generated from these messages.

     ```shell
     git commit -a
     ```
    Note: the optional commit `-a` command line option will automatically "add" and "rm" edited files.

1. Push your branch to GitHub:

    ```shell
    git push origin my-fix-branch
    ```

1. In GitHub, send a pull request to `huggingsound:master`.
* If we suggest changes then:
  * Make the required updates.
  * Re-run the test suites to ensure tests are still passing.
  * Rebase your branch and force push to your GitHub repository (this will update your Pull Request):

    ```shell
    git rebase master -i
    git push -f
    ```

That's it! Thank you for your contribution!

#### After your pull request is merged

After your pull request is merged, you can safely delete your branch and pull the changes
from the main (upstream) repository:

* Delete the remote branch on GitHub either through the GitHub web UI or your local shell as follows:

    ```shell
    git push origin --delete my-fix-branch
    ```

* Check out the master branch:

    ```shell
    git checkout master -f
    ```

* Delete the local branch:

    ```shell
    git branch -D my-fix-branch
    ```

* Update your master with the latest upstream version:

    ```shell
    git pull --ff upstream master
    ```

## <a name="rules"></a> Coding Rules
To ensure consistency throughout the source code, keep these rules in mind as you are working:

* All features or bug fixes **must be tested** by one or more specs (unit-tests).
* All public methods **must be documented** for the final user in some way.
* We follow the [PEP8 Style Guide][pep8-style-guide] for general coding 
  and the [Numpy Docstirng Style Guide][numpy-docstring-style-guide] for code documentation.

## <a name="commit"></a> Commit Message Guidelines

We have very precise rules over how our git commit messages can be formatted. We follow the [Convetional Commit Guide][conventionalcommits]. This leads to **more
readable messages** that are easy to follow when looking through the **project history**.  But also,
we use the git commit messages to **generate the project change log**.

### Commit Message Format
Each commit message consists of a **header**, a **body** and a **footer**.  The header has a special
format that includes a **type** and a **subject**:

```
<type>: <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **header** is mandatory.

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier
to read in various git tools.

The footer should contain a [closing reference to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-issues/linking-a-pull-request-to-an-issue) if any.

```
docs: update changelog to 0.2
```
```
fix: need to depend on latest rxjs and zone.js

The version in our package.json gets copied to the one we publish, and users need the latest of these.

Closes #10
```

### Type
Must be one of the following:

* **release**: Just a release-related commit.
* **build**: Changes that affect the build system or external dependencies
* **ci**: Changes to our CI configuration files and scripts
* **doc**: Documentation only changes
* **feat**: A new feature
* **fix**: A bug fix
* **perf**: A code change that improves performance
* **refact**: A code change that neither fixes a bug, adds a feature nor improves code performance
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
* **test**: Adding missing tests or correcting existing tests
* **revert**: Reverts some previous commit, in this case the reverted commit hash needs to be present in the commit message

### Subject
The subject contains a succinct description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* don't capitalize the first letter
* no dot (.) at the end

### Body
Just as in the **subject**, use the imperative, present tense: "change" not "changed" nor "changes".
The body should include the motivation for the change and contrast this with previous behavior.

### Footer
The footer should contain any information about **Breaking Changes** and is also the place to
reference GitHub issues that this commit **Closes**.

**Breaking Changes** should start with the words `BREAKING CHANGE:` with a space or two newlines. The rest of the commit message is then used for this.

## <a name="dev"></a> Building and Testing

Let's see what needs to be done in your machine before [submit a Pull Request](#submit-pr) 

## Prerequisite Software

Before you can build and test, you must install and configure the
following products on your development machine:

* [Python](https://www.python.org)

* [Git](http://git-scm.com)

* [Poetry](https://python-poetry.org)

## Getting the Sources

Fork and clone the HuggingSound repository:

1. Login to your GitHub account or create one [here](https://github.com).
2. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [HuggingSound
   repository](https://github.com/jonatasgrosman/huggingsound).
3. Clone your fork of the HuggingSound repository and define an `upstream` remote pointing back to
   the HuggingSound repository that you forked in the first place.

```shell
# Clone your GitHub repository:
git clone git@github.com:<github username>/huggingsound.git

# Go to the HuggingSound directory:
cd huggingsound

# Add the HuggingSound repository as an upstream remote to your repository:
git remote add upstream https://github.com/jonatasgrosman/huggingsound.git
```

## Initial config

Clone the environment variables file sample

```shell
cp .env.sample .env
```

After cloning this file, you should set the values of its variables properly

## Installing dependencies

```shell
make setup
```

## Running Tests

```shell
make test
```

**Note**: All the tests are executed on our Continuous Integration infrastructure after a commit push. PRs can only be merged if the code is formatted properly and all tests are passing.

## Linting/verifying your Source Code

(TODO: We need to define this)

[numpy-docstring-style-guide]: https://numpydoc.readthedocs.io/en/latest/format.html
[pep8-style-guide]: https://www.python.org/dev/peps/pep-0008/
[conventionalcommits]: https://www.conventionalcommits.org/en/v1.0.0/
[github-flow]: https://guides.github.com/introduction/flow/