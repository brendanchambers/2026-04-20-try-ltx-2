#!/bin/bash
set -e

echo "Setting up ltx-2-mlx tool..."

# Clone the ltx-2-mlx repository
if [ -d "ltx-2-mlx" ]; then
    echo "ltx-2-mlx directory already exists. Skipping clone."
else
    echo "Cloning ltx-2-mlx repository..."
    git clone https://github.com/dgrauet/ltx-2-mlx.git
    cd ltx-2-mlx

    # Checkout the commit this project was tested with
    echo "Checking out tested commit..."
    git checkout 3962bfce47b91030cf4dab811049f9b825db64e0

    cd ..
fi

echo ""
echo "Setup complete! You can now use the ltx-2-mlx CLI tool."
echo ""
echo "Example usage:"
echo "  uv run ltx-2-mlx generate --prompt 'your prompt here' --output output.mp4"
echo ""
echo "For more information, see: https://github.com/dgrauet/ltx-2-mlx"
