DAT255--Detect-disease-in-chest-X-rays-images

# 🩻 Automatisert deteksjon av lungesykdommer ved bruk av røntgenbilder og kunstig intelligens

---

## 📖 Introduksjon

Dette prosjektet undersøker hvordan dyp læring kan benyttes til å klassifisere lungesykdommer fra røntgenbilder av brystkassen (Chest X-rays). 
Formålet er å utvikle en modell som kan identifisere patologiske funn i medisinske bilder og gi sannsynlighetsestimater for flere diagnoser samtidig (multi-label klassifisering).

Prosjektet er gjennomført som en del av emnet **DAT255 – Deep Learning**.


## 🎯 Målsetting

Prosjektets hovedmål er å:

Utvikle og trene en konvolusjonsnevralt nettverksmodell (CNN) fra bunnen av (tilfeldig initialiserte vekter). 
Sammenligne ytelsen med mer avanserte arkitekturer som ResNet og DenseNet, samt eventuelt en Vision Transformer. 
Analysere hvordan modellarkitektur påvirker klassifiseringsytelse i medisinsk bildeanalyse. 
Evaluere modellene ved hjelp av relevante metrikker for medisinsk klassifisering.

## 📂 Datasett

Vi benytter **CheXpert-datasettet** fra Stanford ML Group:

https://stanfordmlgroup.github.io/competitions/chexpert/

Datasettet inneholder:

- 224 316 røntgenbilder  
- 14 ulike medisinske tilstander  

Hver tilstand er merket som:

- `1` → Positiv  
- `0` → Negativ  
- `-1` → Usikker  


## 🧠 Modellarkitektur

Følgende modeller inngår i prosjektet:

- Egendefinert CNN (trent fra bunnen av)
- ResNet (transfer learning)
- DenseNet
- Vision Transformer


## 👥 Gruppemedlemmer

- Torstein Sleire  
- Mathilde Røssland Johnsen  
- Astrid Rødland  
