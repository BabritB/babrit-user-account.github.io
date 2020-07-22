---
title: "Easy ML Model Deployment"
date: 2020-07-12
tags: [model deployment]
header:
  image: "/images/model-deployment.png"
excerpt: "Deployment, Heroku"
mathjax: "true"
---
# Deployment on Heroku

#### Heroku is one of the first cloud platform as a service (PaaS) and supports several languages - Ruby, Java, Node.js, Scala, Clojure, Python, PHP, and Go.
The first thing we need to do is define which libraries our application uses. That way, Heroku knows which ones to provide for us, similar to how we install them locally when developing the app.To perform this task we need to have a requirement.txt file where all the required libraries will be mentioned including their versions.

```python
$ pip freeze > requirements.txt
```


```python
Flask==1.1.1
gunicorn==19.9.0
itsdangerous==1.1.0
Jinja2==2.10.1
MarkupSafe==1.1.1
Werkzeug==0.15.5
numpy>=1.9.2
scipy>=0.15.1
scikit-learn>=0.18
matplotlib>=1.4.3
pandas>=0.19
```

#### Pay attention to the misspelling 'requirements'
For Heroku to be able to run our application like it should, we need to define a set of processes/commands that it should run beforehand. These commands are located in the Procfile:

```python
web: gunicorn app:app
```
The web command tells Heroku to start a web server for the application, using gunicorn. Since our application is called app.py, we've set the app name to be app as well.
### Create Heroku Account


```python
Sign Up -> https://signup.heroku.com/
```

#### Once that is out of the way, on the dashboard, select New -> Create new app:

Once the application is created on Heroku, we're ready to deploy it online.

### GIT : 

To upload our code, we'll use Git. First, let's make a git repository:


```python
$ git init .
```
And now, we add our files and commit:

```python
$ git add app.py Procfile requirements.txt
$ git commit -m "first commit"
```

### Deploying the App to Heroku:
For the final deploy , we'll need to install the Heroku CLI with which we'll run Heroku-related commands. Let's login to our account using our credentials by running the command:

```python
$ heroku login -i
```
Alternatively, we can login using the browser if we run the command:

```python
$ heroku login
```
At this point, while logged in, we should add our repository to the remote one:

```python
$ heroku git:remote -a {your-project-name}
```
Remember : replace {your-project-name} with the actual name of your project you selected in the earlier step.Now let's push the git to upload on Heroku. 

```python
$ git push heroku master
```
A Process log will be displayed over there:

```python
...
remote: -----> Discovering process types
remote:        Procfile declares types -> web
remote:
remote: -----> Compressing...
remote:        Done: 45.1M
remote: -----> Launching...
remote:        Released v4
remote:        https://{your-project-name}.herokuapp.com/ deployed to Heroku
remote:
remote: Verifying deploy... done.
To https://git.heroku.com/{your-project-name}.git
   ae85864..4e63b46  master -> master
```
Now you have successfully uploaded your first web app to Heroku! It's now time now to test and verify our app.
### Testing:
In the log that has been shown in the console you will find a link for your application https://{your-project-name}.herokuapp.com/, this link can also be found under the Settings tab, in the Domains and certificates section:

```python
https://{your-project-name}.herokuapp.com/
```
You can also cross verify with Heroku interface:

```python

```
