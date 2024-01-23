<a name="readme-top"></a>
<!-- https://github.com/othneildrew/Best-README-Template/tree/master -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/SamuReyes/WindViViT">
    <!-- <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
  </a>

  <h3 align="center">WindViViT</h3>

  <p align="center">
    🚧 Project under construction 🚧
    <br />
    <br />
    <a href="https://github.com/SamuReyes/WindViViT/issues">Report Bug</a>
    ·
    <a href="https://github.com/SamuReyes/WindViViT/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project


<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started


### Prerequisites


### Installation


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
This section provides guidance on how to get your local machine set up to run the project. By following these instructions, you'll have a local copy of the project up and running.

### Prerequisites
Before you begin, ensure you have met the following requirements:

- Docker: You need Docker installed on your system. If you haven't already done so, download and install Docker for your operating system from [Docker's official website](https://www.docker.com/).
- NVIDIA Drivers (Optional): If you plan to leverage GPU capabilities, ensure you have the [NVIDIA drivers](https://www.nvidia.com/download/index.aspx) installed for Docker to access the GPU.
- Git (Optional): To clone the project repository, you might need Git installed. Download it from [Git's official website](https://git-scm.com/).

### Installation
To install the project, follow these steps:

**Clone the Repository:** 

```bash
git clone https://github.com/SamuReyes/WindViViT
```

**Configure Docker Compose:**

Navigate to the /docker directory and **modify the volume path** in the docker-compose.yml file to match your local project directory. Additionally, you can modify the name of the container that will be created.

**Build the Docker Image:**

In the /docker directory, run the following command to build the Docker image. Replace <user> and <password> with your desired credentials:

```bash
sudo docker build -t docker-pytorch2.0.1:wind-prediction -f Dockerfile.pytorch-wind --build-arg USER=<user> --build-arg PASSWORD=<password> .
```

**Start the Docker Container:**

```bash
docker-compose up -d
```

**Accessing the Services:**

"The project is now running inside Docker containers. You can access Jupyter Notebook at http://localhost:8888. 
Additionally, for a more integrated development experience, you can attach to these Docker containers using Visual Studio Code, installing with the 'Remote - Containers' and 'Docker' extensions. 
For direct command-line interactions, use:

```bash
docker exec -it [container_name] /bin/bash
```

By following these steps, you should have the project running on your local machine. If you encounter any issues, please refer to the Docker and Docker Compose documentation for troubleshooting tips.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Processing and Normalization
- [x] EDA
- [ ] Build the model
    - [ ] Patch embedding
    - [ ] Attention
    - [ ] ...


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Samuel Reyes - [LinkedIn](https://www.linkedin.com/in/samuel-reyes-sanz/) - samuel.reyes.sanz@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

<!-- [Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/-->
