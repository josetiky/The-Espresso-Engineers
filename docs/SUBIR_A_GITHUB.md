# Cómo subir el proyecto a GitHub

> El repositorio **debe ser PÚBLICO** (un repo privado = descalificación, regla
> 3.7.4). Fecha límite: **19 de junio de 2026, 23:59**. Después no se permiten cambios.

⚠️ **Antes de subir:** los videos (`.mp4`) y los respaldos de modelos ya están en
`.gitignore` porque pesan demasiado para GitHub (límite 100 MB por archivo). El
video demo y el Reel se **enlazan** en el README (YouTube/Drive/Instagram).

---

## Opción A — GitHub Desktop (la más fácil, recomendada)

1. Descarga e instala **GitHub Desktop**: https://desktop.github.com
2. Inicia sesión con tu cuenta de GitHub (crea una en https://github.com si no tienes).
3. Menú **File → Add Local Repository…** y elige la carpeta `futbotmx`.
   - Si dice que no es un repositorio, clic en **“create a repository”**.
4. Escribe un resumen en "Summary" (p. ej. `Proyecto FutBotMX`) y clic en
   **Commit to main**.
5. Arriba, clic en **Publish repository**.
   - **DESMARCA** la casilla *"Keep this code private"* → debe quedar **público**.
6. Listo. Clic en **View on GitHub** para ver tu repo en línea.

---

## Opción B — Terminal (git)

### 1. Crea el repositorio vacío en GitHub
- Entra a https://github.com/new
- Nombre: `futbotmx` (o el que quieras). Visibilidad: **Public**.
- **No** marques "Add a README" (ya tienes uno).
- Clic en **Create repository** y copia la URL (algo como
  `https://github.com/TU_USUARIO/futbotmx.git`).

### 2. Sube el proyecto (en la terminal, dentro de la carpeta)
```bash
cd ~/Desktop/futbotmx
git init
git add .
git commit -m "Proyecto Copa FutBotMX - Vision por Computadora"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/futbotmx.git
git push -u origin main
```

### 3. Autenticación
Al hacer `push`, GitHub pedirá usuario y contraseña. **La contraseña ya no
funciona**: necesitas un **token de acceso personal**:
- Ve a https://github.com/settings/tokens → **Generate new token (classic)** →
  marca el permiso **repo** → genera y copia el token (`ghp_...`).
- Cuando `git push` pida la contraseña, **pega el token** (no se ve al pegarlo).

> Si esto se complica, usa la **Opción A (GitHub Desktop)**, que maneja el login solo.

---

## Después de subir (verificación)

- [ ] Abre tu repo en una **ventana de incógnito** (sin sesión) para confirmar
      que es **público** y se ve.
- [ ] Revisa que estén: `README.md`, `LICENSE`, el código, `docs/`, `utils/`,
      `requirements.txt` y el modelo `futbotmx_v2.pt`.
- [ ] Pega en el `README.md` el **enlace al Reel de Instagram** y al **video demo**
      (súbelo a YouTube/Drive como "no listado/público" y enlázalo).
- [ ] Registra el **enlace del repositorio** en el formulario oficial del reto.

---

## Si un archivo es demasiado grande (error al hacer push)

Si el `push` falla por un archivo > 100 MB:
```bash
# Quítalo del seguimiento (sigue en tu disco, solo no se sube)
git rm --cached "NOMBRE_DEL_ARCHIVO"
echo "NOMBRE_DEL_ARCHIVO" >> .gitignore
git commit -m "Excluir archivo grande"
git push
```
Para versionar pesos grandes hay **Git LFS** (https://git-lfs.com), pero para el
reto basta con incluir el código + `futbotmx_v2.pt` (19 MB) y enlazar los videos.

---

## ¿Quieres subir el video demo al repo de todas formas?

Si tu demo pesa **menos de 100 MB** y quieres incluirlo (aunque esté en
`.gitignore`), fuérzalo:
```bash
git add -f "Video Comentado.mp4"
git commit -m "Agregar video demo"
git push
```
Aun así, lo más limpio es **enlazarlo** (YouTube/Drive) en el README.
