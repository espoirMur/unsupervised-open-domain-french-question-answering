# multilingual-drc-news-chatbot


[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

## Table of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

<!-- ABOUT THE PROJECT -->

## About The Project

_Multilingual Drc news chatbot_

This project is a chatbot project that will be answering question about Congolese news.

<!-- Build with -->

## Built With

- [Python](https://www.python.org/)
- [scrapy](https://scrapy.org)
- [scrapydweb](https://github.com/my8100/scrapydweb)
- [Github Actions](https://github.com/features/actions)

<!-- GETTING STARTED -->

## Getting started

Clone the project to have a local copy in your machine.

We have decided to use docker to build and have the project running...

### Install Postgres

The project also use postgres as database , install it and create a database for the project. Keep it's name somewhere for future use.

Follow this [URL](https://medium.com/coding-blocks/creating-user-database-and-adding-access-on-postgresql-8bfcd2f4a91e) to create a user and a database

### Generate the .env file

`cp .env.sample .env`

### Run Docker

Make sure you have docker installed and running and docker-compose and then go inside the project directory and run :

`docker-compose up -d --build`

After running the docker container, you will find the scrapydweb dashoard for the monitoring and the scheduling of the scrapers at :

`localhost:5000`


<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->

## License

<!-- CONTACT -->

## Contact

Espoir Murhabazi - [Twitter](https://twitter.com/esp_py) - espoir.mur on gmail

Project Link: [https://github.com/espoirMur/multilingual-drc-news-chatbot](https://github.com/espoirMur/multilingual-drc-news-chatbot)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/espoirMur/multilingual-drc-news-chatbot/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/espoirMur/multilingual-drc-news-chatbot/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/espoirMur/balobi_nini/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/espoirMur/multilingual-drc-news-chatbot/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/espoirMur/multilingual-drc-news-chatbot/blob/master/LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
