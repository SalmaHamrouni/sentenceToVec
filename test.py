from lib.sentence2vec import Sentence2Vec

model = Sentence2Vec('./data/salma-offres-vectors.model')

# turn job title to vector
print(model.get_vector('Développeur Angular Très bon niveau sur les technologies actuelles WEB (Java, Angular) Capacité à appréhender et à travailler dans un code existant Environnement technique Angular 11, Typescript Node NiceToHave : RxJs  NgRx Intégration d’APIJIRA  Github'))

# not similar job
print(model.similarity('Développeur Angular Très bon niveau sur les technologies actuelles WEB (Java, Angular) Capacité à appréhender et à travailler dans un code existant Environnement technique Angular 11, Typescript Node NiceToHave : RxJs  NgRx Intégration d’APIJIRA  Github',
                       'Cherni - Développeur Angular Angular ANGULARJS TypeScript JAVASCRIPT HTML5 CSS NODE.JS Développeur Sénior Front Au sein de la DSI-RC / Entreprise Edition DSN, participer aux développements de refonte des IHMs et du parcours utilisateur des gestionnaires afin d’améliorer la productivité des traitements des dossiers et faciliter les interlocutions avec les entreprises, une application web fournissant différents écrans de synthèse permettant une première analyse pertinente. Réalisations Développement FRONT Angular 10 TypeScript (développement de composants web génériques) Environnement : Angular 10, npm, Typescript, TSlint, Rxjs, Angular Material, SCSS, Html, Zeplin, SVN, Jira, Squirrel-Sql, Jenkins, Scrum'))

# a bit similar job
print(model.similarity('Développeur Angular Très bon niveau sur les technologies actuelles WEB (Java, Angular) Capacité à appréhender et à travailler dans un code existant Environnement technique Angular 11, Typescript Node NiceToHave : RxJs  NgRx Intégration d’APIJIRA  Github',
                       'Guilherme - Développeur PYTHON PYTHON POSTGRES DJANGO'))

# similar job
print(model.similarity('Développeur Angular Très bon niveau sur les technologies actuelles WEB (Java, Angular) Capacité à appréhender et à travailler dans un code existant Environnement technique Angular 11, Typescript Node NiceToHave : RxJs  NgRx Intégration d’APIJIRA  Github',
                       'Polo - manager'))
