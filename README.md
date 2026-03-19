# DAT255--Detect-disease-in-chest-X-rays-images

🩻 Automatisert deteksjon av lungesykdommer ved bruk av røntgenbilder og kunstig intelligens

📖 Introduksjon

Dette prosjektet undersøker hvordan dyp læring kan benyttes til å klassifisere lungesykdommer fra røntgenbilder av brystkassen (Chest X-rays). Formålet er å utvikle en modell som kan identifisere patologiske funn i medisinske bilder og gi sannsynlighetsestimater for flere diagnoser samtidig (multi-label klassifisering).

Prosjektet er gjennomført som en del av emnet DAT255 – Deep Learning.

🎯 Målsetting

Utvikle og trene en konvolusjonsnevralt nettverksmodell (CNN) fra bunnen av (tilfeldig initialiserte vekter).
Sammenligne ytelsen med mer avanserte arkitekturer som ResNet og DenseNet, samt eventuelt en Vision Transformer.
Analysere hvordan modellarkitektur påvirker klassifiseringsytelse i medisinsk bildeanalyse.
Evaluere modellene ved hjelp av relevante metrikker for medisinsk klassifisering.

📂 Datasett

Vi benytter CheXpert-datasettet fra Stanford ML Group:
https://stanfordmlgroup.github.io/competitions/chexpert/

<<<<<<< HEAD
MEN bruker datasettet som tar mindre plass fra Kaggle:

https://www.kaggle.com/datasets/ashery/chexpert

Datasettet inneholder:

Datasettet inneholder 224 316 røntgenbilder vurdert for 14 ulike medisinske tilstander. Hver tilstand er annotert som:
Positiv (1)
Negativ (0)
Usikker (-1)
Datasettet er ikke inkludert i dette repositoriet grunnet størrelse og lisenskrav.

🧠 Modellarkitektur

Følgende modeller inngår i prosjektet:
Egendefinert CNN (trent fra bunnen av)
ResNet (transfer learning)
DenseNet
Vision Transformer (valgfritt)
Modellene implementeres og trenes ved bruk av PyTorch eller TensorFlow.


- Egendefinert CNN (trent fra bunnen av)
- ResNet (transfer learning)
- DenseNet
- Vision Transformer


## 👥 Gruppemedlemmer

- Torstein Hosøy Sleire  
- Mathilde Røssland Johnsen  
- Astrid Rødland  

👥 Gruppemedlemmert
Torstein Sleire
Mathilde Røssland Johnsen
Astrid Rødland

