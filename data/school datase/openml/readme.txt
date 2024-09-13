# Jeu de données sur les performances académiques des étudiants (xAPI-Edu-Data)

## Caractéristiques du jeu de données
- Type : Multivarié
- Nombre d'instances : 480
- Domaine : E-learning, Éducation, Modèles prédictifs, Exploration de données éducatives
- Caractéristiques des attributs : Entier/Catégorique
- Nombre d'attributs : 16
- Date : 8 novembre 2016
- Tâches associées : Classification
- Valeurs manquantes ? Non
- Format de fichier : xAPI-Edu-Data.csv

## Source
Elaf Abu Amrieh, Thair Hamtini et Ibrahim Aljarah, Université de Jordanie, Amman, Jordanie
http://www.Ibrahimaljarah.com
www.ju.edu.jo

## Informations sur le jeu de données
Ce jeu de données éducatif a été collecté à partir d'un système de gestion de l'apprentissage (LMS) appelé Kalboard 360. Kalboard 360 est un LMS multi-agents conçu pour faciliter l'apprentissage grâce à l'utilisation de technologies de pointe. Ce système offre aux utilisateurs un accès synchrone aux ressources éducatives depuis n'importe quel appareil connecté à Internet.

Les données ont été collectées à l'aide d'un outil de suivi des activités des apprenants appelé Experience API (xAPI). L'xAPI est un composant de l'architecture de formation et d'apprentissage (TLA) qui permet de suivre les progrès d'apprentissage et les actions des apprenants, comme la lecture d'un article ou le visionnage d'une vidéo de formation.

Le jeu de données comprend 480 enregistrements d'étudiants et 16 caractéristiques. Les caractéristiques sont classées en trois catégories principales :
1. Caractéristiques démographiques (ex : sexe et nationalité)
2. Caractéristiques du parcours académique (ex : niveau d'études, niveau de classe et section)
3. Caractéristiques comportementales (ex : lever la main en classe, consulter les ressources, répondre à l'enquête par les parents et satisfaction scolaire)

## Attributs
1. Gender : sexe de l'étudiant (nominal : 'Male' ou 'Female')
2. Nationality : nationalité de l'étudiant (nominal : Kuwait, Lebanon, Egypt, SaudiArabia, USA, Jordan, Venezuela, Iran, Tunis, Morocco, Syria, Palestine, Iraq, Lybia)
3. Place of birth : lieu de naissance de l'étudiant (nominal : Kuwait, Lebanon, Egypt, SaudiArabia, USA, Jordan, Venezuela, Iran, Tunis, Morocco, Syria, Palestine, Iraq, Lybia)
4. Educational Stages : niveau d'études de l'étudiant (nominal : lowerlevel, MiddleSchool, HighSchool)
5. Grade Levels : niveau de classe de l'étudiant (nominal : G-01 à G-12)
6. Section ID : section de l'étudiant (nominal : A, B, C)
7. Topic : sujet du cours (nominal : English, Spanish, French, Arabic, IT, Math, Chemistry, Biology, Science, History, Quran, Geology)
8. Semester : semestre de l'année scolaire (nominal : First, Second)
9. Parent responsible for student : parent responsable de l'étudiant (nominal : mom, father)
10. Raised hand : nombre de fois où l'étudiant a levé la main en classe (numérique : 0-100)
11. Visited resources : nombre de fois où l'étudiant a consulté le contenu du cours (numérique : 0-100)
12. Viewing announcements : nombre de fois où l'étudiant a consulté les nouvelles annonces (numérique : 0-100)
13. Discussion groups : nombre de fois où l'étudiant a participé à des groupes de discussion (numérique : 0-100)
14. Parent Answering Survey : le parent a répondu ou non aux enquêtes fournies par l'école (nominal : Yes, No)
15. Parent School Satisfaction : degré de satisfaction des parents envers l'école (nominal : Yes, No)
16. Student Absence Days : nombre de jours d'absence pour chaque étudiant (nominal : above-7, under-7)

## Classification des étudiants
Les étudiants sont classés en trois intervalles numériques basés sur leur note/marque totale :
- Niveau bas : intervalle incluant les valeurs de 0 à 69
- Niveau moyen : intervalle incluant les valeurs de 70 à 89
- Niveau élevé : intervalle incluant les valeurs de 90 à 100

## Articles pertinents
1. Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). Mining Educational Data to Predict Students' Academic Performance using Ensemble Methods. International Journal of Database Theory and Application, 9(8), 119-136.
2. Amrieh, E. A., Hamtini, T., & Aljarah, I. (2015, November). Preprocessing and analyzing educational data set using X-API for improving student's performance. In Applied Electrical Engineering and Computing Technologies (AEECT), 2015 IEEE Jordan Conference on (pp. 1-5). IEEE.

## Demande de citation
Veuillez inclure ces citations si vous prévoyez d'utiliser ce jeu de données :

1. Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016). Mining Educational Data to Predict Students' Academic Performance using Ensemble Methods. International Journal of Database Theory and Application, 9(8), 119-136.
2. Amrieh, E. A., Hamtini, T., & Aljarah, I. (2015, November). Preprocessing and analyzing educational data set using X-API for improving student's performance. In Applied Electrical Engineering and Computing Technologies (AEECT), 2015 IEEE Jordan Conference on (pp. 1-5). IEEE.