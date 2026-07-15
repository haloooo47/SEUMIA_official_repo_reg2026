#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SAVE_DIR="${REG2_SAVE_DIR:-${SCRIPT_DIR}}"
mkdir -p "$SAVE_DIR"
MODEL_DIR="${REG2_MODEL_DIR:-${SCRIPT_DIR}/models}"

# Set default container name
DOCKER_IMAGE_TAG="reg2026_algorithm"

echo ""
echo "= STEP 0 = Checking packaged model assets"
REQUIRE_PACKAGE_DEFAULT="${REG2_REQUIRE_PACKAGE_DEFAULT_PROFILE:-1}"
if [ "${REG2_ALLOW_NO_PACKAGE_PROFILE:-0}" = "1" ]; then
    REQUIRE_PACKAGE_DEFAULT=0
fi
if [ "$REQUIRE_PACKAGE_DEFAULT" = "1" ]; then
    if [ ! -f "${MODEL_DIR}/reg2_ensemble_profile/package_default.json" ] && \
       [ ! -f "${MODEL_DIR}/reg2_ensemble_profile/default.json" ]; then
        echo "Error: package parity guard requires models/reg2_ensemble_profile/package_default.json or default.json"
        echo "       Provide the production profile first, or set REG2_ALLOW_NO_PACKAGE_PROFILE=1 to package without one."
        exit 1
    fi
fi
PROFILE_ARGS=()
if [ -n "${REG2_PACKAGE_PROFILE:-}" ]; then
    PROFILE_ARGS=(--profile "${REG2_PACKAGE_PROFILE}")
fi
CHECK_ARGS=(--model-root "${MODEL_DIR}" "${PROFILE_ARGS[@]}")
if [ "$REQUIRE_PACKAGE_DEFAULT" = "1" ]; then
    CHECK_ARGS+=(--require-package-default)
fi
python3 "${SCRIPT_DIR}/scripts/check_submission_model_assets.py" \
    "${CHECK_ARGS[@]}"
echo "==== Done"
echo ""

echo ""
echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"
echo "==== Done"
echo ""

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG" )

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

# Format the build information to remove special characters
formatted_build_info=$(echo $build_timestamp | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

# Set the output filename with timestamp and build information
output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SAVE_DIR}/$output_filename"

# Save the Docker-container image and gzip it
echo "= STEP 2 = Saving the image"
echo "This can take a while."

docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_path"
printf "Saved as: \e[32m${output_path}\e[0m\n"

echo "==== Done"
echo ""


# Create the tarbal
echo "= STEP 3 = Packing the model"
echo "This can take a while."
output_tarball_name="${SAVE_DIR}/model.tar.gz"

tar -czf "$output_tarball_name" -C "${MODEL_DIR}" .
printf "Saved as: \e[32m${output_tarball_name}\e[0m\n"

echo "==== Done"
echo ""

printf "\e[33mNext steps:\e[0m\n"
printf "  1. Upload \e[32m%s\e[0m  →  Grand Challenge > Algorithm > Container images\n" "$output_path"
printf "  2. Upload \e[32m%s\e[0m           →  Grand Challenge > Algorithm > Models\n" "$output_tarball_name"
printf "\e[31mIMPORTANT: Upload model.tar.gz as a separate Model on your Algorithm (not inside the image).\e[0m\n"
