![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)


# CS 180 Project, Group KEY: Sugarcane Image Classification

The documentation for the training process is available in `model_training.ipynb`

The demo version is available in `model_demo.ipynb`.

The following research papers were referred to during development
- [Vision Transformer-Based Framework for AI-Generated Image Detection in Interior Design](https://informatica.si/index.php/informatica/article/view/7979#:~:text=The%20best%20tradeoff%20between%20accuracy,with%20lower%20accuracy%20than%20speed)
- [Reducing Complexity of 3D Indoor Object Detection](https://www.researchgate.net/publication/329466744_Reducing_Complexity_of_3D_Indoor_Object_Detection)
- [Optimization of vision transformer-based detection of lung diseases from chest X-ray images](https://pmc.ncbi.nlm.nih.gov/articles/PMC11232177/)

## Running the Web_App

This project and its dependencies are managed using [uv by Astral](https://docs.astral.sh/uv/).

To run the project, follow the steps below.
1. Clone the repo
2. Install [uv by Astral](https://docs.astral.sh/uv/).
3. If Python 3.12 is not available within the system, run `uv python install 3.12`
4. Run `uv venv -p 3.12`
5. Run `uv sync`
6. Change the current active directory to `web`
7. Run `uv run uvicorn web_app.asgi:application`
No need to collectstatic, since the staticfiles are included in the repository due to their small size.

For development, consider the following steps after *Step 6*.
- Ensure that [NodeJS](https://nodejs.org/en/download) is installed
- Run `python manage.py tailwind install`

