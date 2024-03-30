# `safe_eval` Job Set Up

## Building `safe_eval` Docker Image

To build Docker image `cd` to this directory and run:

```bash
docker build -t safe_eval .
```

## (Optional) Running `safe_eval` Docker Image

Once built, to run the Docker image run:

```bash
docker run --rm safe_eval "$code" "$test_code"
```

where `$code` and `$test_code$` are bash variables with your code and tests respectively in string form.

If there are no issues, the Docker image will return a JSON formatted string with the results to stdout.

## Pushing (Uploading) Docker Image to Google Cloud

1. Folow the steps [here](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images) to set up a Docker repository on Google Cloud

2. Tag the Docker image with the full path to your Google Cloud Docker repository:

```bash
DOCKER_REPO="GCLOUD_REGION-docker.pkg.dev/GCLOUD_PROJECT_NAME/DOCKER_REPO_NAME"
docker tag safe_eval $DOCKER_REPO/safe_eval
```

where `DOCKER_REPO` is the full Google Cloud path to your Google Cloud Docker repository. This can be found and copied from the repositories page on Google Cloud.

3. Push the Image to Google Cloud

```bash
docker push $DOCKER_REPO/safe_eval
```

where `DOCKER_REPO` is the same as above.

### WARNING: If you built the Docker image on a Apple Silicon Mac, you can upload the image, but it will NOT be able to run on Google Cloud

This is because the Docker image was compiled under a different system architecture (`arm64` instead of the required `amd64`).

To fix this you must [Build and push a Docker image with Cloud Build](https://cloud.google.com/build/docs/build-push-docker-image), i.e. push the `Dockerfile` and have Google Cloud build it. To do this:
 1. Make sure you've completed the [Before you begin](https://cloud.google.com/build/docs/build-push-docker-image#before-you-begin) steps and have a Docker repository set up (see #1 at the beginning of this section)
 2. Run:

```bash
DOCKER_REPO="GCLOUD_REGION-docker.pkg.dev/GCLOUD_PROJECT_NAME/DOCKER_REPO_NAME"
gcloud builds submit --region=GCLOUD_REGION --tag $DOCKER_REPO/safe_eval
```

where `DOCKER_REPO` is the full Google Cloud path to your Google Cloud Docker repository. This can be found and copied from the repositories page on Google Cloud.

## Creating a Google Cloud Run Job from the Docker Image

1. Go to the [Google Cloud Run](https://console.cloud.google.com/run/jobs) service in Google Cloud

2. Click `Create Job`

3. Click `Select` and find the Docker Image pushed to the Google Cloud Artifact (Docker) Registry above

4. Click `Select` again to confirm

5. If needed, give the job a name

6. You can use the default region (`us-central1 (Iowa)`)

7. Under the `Container, Variables & Secrets, Connections, Security` section `General` tab set `Number of retries per failed task` to `0`

8. Click `Create`