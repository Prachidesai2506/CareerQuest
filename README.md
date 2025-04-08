# QareerQuest

**AI-Powered Interview Preparation & Hiring Platform**

QareerQuest is a smart, AI-driven web application designed to help **students** ace their interviews and assist **companies** in hiring the best talent. With interactive mock tests, AI-based interview modules, resume parsing, and real-time coding compilers, QareerQuest aims to revolutionize how job seekers prepare and companies hire.

---

## Key Features

### For Students:
- Secure Registration & Login
- Mock Aptitude Tests
- Subject-Specific Quizzes
- AI-Driven Mock Interviews with detailed performance analysis
- Resume Parser for auto-filled profile creation
- Confidence Detector analyzing voice tone, facial expressions, and eye movement
- Built-in Coding Compiler for hands-on coding practice
- Detailed Interview Reports with strengths and areas of improvement

### For Companies:
- Create Job Listings with Quizzes
- AI-Based Interview Scheduling and Evaluation
- Access Parsed Resumes & Candidate Profiles
- Data-Driven Hiring Decisions

---

## User Stories

- As a student, I want to give mock and AI-driven interviews so I can improve my skills and understand my readiness.
- As a company, I want to conduct AI interviews and quizzes to evaluate candidates efficiently.
- As a job seeker, I want to easily set up my profile through resume parsing.

---

## System Modules

| Module                       | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| User Registration/Login     | Allows students and companies to securely sign up and log in               |
| Mock Test Engine            | Offers aptitude and subject-specific quizzes                               |
| AI Interview Module         | Conducts interviews and evaluates responses using AI models                |
| Confidence Detection        | Analyzes behavioral cues during interviews                                 |
| Resume Parser               | Extracts data from resumes for easy profile setup                          |
| Job Listings & Quizzes      | Companies can create job posts and evaluate applicants                     |
| Coding Practice Platform    | Built-in compiler to test programming skills                               |
| Interview Report Generator  | Generates detailed reports and feedback after AI interviews                |

---

## System Architecture

- Frontend: Web-based user interface
- Backend: Handles logic, data processing, and AI model integration
- Database: Stores user data, test questions, results, resumes
- AI Models: Used for interview response evaluation and confidence detection

---

## Requirements

### Functional
- User registration and login
- Mock test participation
- AI interview participation and evaluation
- Resume parsing
- Job posting & quiz setup (company)
- Coding practice with output
- Confidence score generation

---

## Deployment Considerations

- Lightweight design with minimal dependencies
- No cost cloud dependency
- Scalable to support high traffic during peak times

---

## Authors

- Kaushal Bhanderi - 22CE005  
- Prachi Desai - 22CE025  
- Mit Monpara - 22CE070  
- Jalay Movaliya - 22CE071  

---

## Institution

U & P U. Patel Department of Computer Engineering  
Charotar University of Science and Technology

First of all create one virtual environment using this command 
- ```python -m venv env```

then activate that virtual environment and then install requirement.txt using this command
- ```pip install -r requirements.txt```

then after to run uvicorn server run this command
- ```uvicorn app:app --reload```
