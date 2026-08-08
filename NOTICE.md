# Mentions légales et composants tiers

Le README annonçait une licence MIT alors qu'aucun fichier `LICENSE` n'existait
et que le dépôt redistribue Stockfish, publié sous **GPL v3**. Ce document
clarifie la situation.

## ChessMate

Code source propre au projet — `main.py`, `vision.py`, `classifier.py`,
`fen.py`, `engine.py`, `llm.py`, `worker.py`, `autoplay.py`, `overlay.py`,
`utils.py`, `i18n.py`, `config.py`, `paths.py`, `train_resnet.py`, `tests/`,
`lang/`, `assets/` — sous **licence MIT** (voir `LICENSE`).

## Stockfish — `engines/`

- **Licence :** GNU General Public License version 3 (texte intégral dans
  `engines/Copying.txt`).
- **Site :** <https://stockfishchess.org> · **Source :**
  <https://github.com/official-stockfish/Stockfish>
- **Contenu redistribué :** le code source complet (`engines/src/`), la
  documentation (`engines/wiki/`) et un binaire Windows précompilé.

### Pourquoi les deux licences coexistent

ChessMate ne se lie pas à Stockfish : il le lance comme **processus séparé** et
dialogue avec lui via le protocole texte UCI, sur des tubes standard. Il ne
s'agit donc pas d'une œuvre dérivée au sens de la GPL, et la distribution
conjointe relève de l'« agrégat » explicitement autorisé par la section 5 de la
GPL v3 :

> A compilation of a covered work with other separate and independent works […]
> in or on a volume of a storage or distribution medium, is called an
> "aggregate" if the compilation and its resulting copyright are not used to
> limit the access or legal rights of the compilation's users […]. Inclusion of
> a covered work in an aggregate does not cause this License to apply to the
> other parts of the aggregate.

### Obligations si vous redistribuez ChessMate

En diffusant ce dépôt (ou un fork, ou un binaire construit à partir de lui)
avec `engines/` :

1. **conserver** `engines/Copying.txt` et les mentions de copyright de
   Stockfish ;
2. **fournir le code source** de la version de Stockfish distribuée — c'est
   déjà le cas via `engines/src/` ; ne supprimez pas ce dossier en conservant
   le `.exe` ;
3. **signaler** toute modification apportée aux sources de Stockfish ;
4. ne pas ajouter de restriction contredisant la GPL sur cette partie.

Si vous préférez éviter ces obligations, retirez `engines/` du dépôt et laissez
l'utilisateur installer Stockfish lui-même : ChessMate le cherche déjà dans le
`PATH` et via la variable d'environnement `STOCKFISH_PATH`.

## Autres composants

| Composant | Licence | Rôle |
|-----------|---------|------|
| PySide6 (Qt for Python) | LGPL v3 | interface graphique |
| python-chess | GPL v3 *(bibliothèque Python, liaison dynamique)* | règles du jeu, protocole UCI |
| PyTorch / torchvision | BSD-3-Clause | classifieur de cases |
| OpenCV | Apache 2.0 | détection du plateau |
| Pillow, NumPy, mss, requests, PyAutoGUI | MIT / BSD / HPND | utilitaires |

> **À noter :** `python-chess` est lui-même sous GPL v3. Si vous distribuez un
> exécutable *packagé* (PyInstaller, Nuitka…) embarquant python-chess, cet
> exécutable devient une œuvre combinée et doit être distribué sous GPL v3.
> Distribuer ChessMate sous forme de code source, comme aujourd'hui, ne pose
> pas ce problème : chaque dépendance conserve sa propre licence.

## Poids du modèle

`resnet18_chess.pt` est un ResNet-18 pré-entraîné sur ImageNet (poids
torchvision, BSD-3-Clause) puis affiné sur le contenu de `dataset/`. Il est
diffusé sous les mêmes termes que ChessMate (MIT).

## Usage

ChessMate est un outil **pédagogique**. L'utiliser pour obtenir une assistance
non autorisée dans une partie classée enfreint les conditions d'utilisation de
chess.com, lichess et de la quasi-totalité des plateformes. Ni le projet ni ses
contributeurs ne sauraient être tenus responsables d'un tel usage.
