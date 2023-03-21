FROM python:3.10-slim

# Define pluggable args
ARG APP_DIR=/opt/igad-mesa
ARG USERNAME=mesa
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG PACKAGES_FILE=requirements.txt

# set env vars
ENV APP_DIR=${APP_DIR} DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 

# Install system packages and configire app home directory.
RUN apt-get update --yes --quiet && apt-get install --yes --quiet --no-install-recommends --fix-missing \
    gdal-bin libgdal-dev libproj-dev libgeos-dev \
    && mkdir -p ${APP_DIR} && \
    groupadd --system --gid ${GROUP_ID} ${USERNAME} && \
    useradd --system --gid ${GROUP_ID} --uid ${USER_ID} \ 
    --no-create-home --home-dir ${APP_DIR} ${USERNAME} && \
    chown -Rf ${USER_ID}:${GROUP_ID} ${APP_DIR}
 
# switch user
USER ${USERNAME}

# set working directory
WORKDIR ${APP_DIR}

# set env var
ENV PATH=${PATH}:${APP_DIR}/.local/bin

# copy project code
COPY . ${APP_DIR}/

# install dependencies
RUN pip install pip --upgrade && \
    pip install -r ${PACKAGES_FILE}

CMD ["mesa", "runserver"]