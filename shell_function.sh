# Add this to your ~/.bashrc or ~/.zshrc for easy environment activation
# Copy and paste the function below into your shell profile

practice_env() {
    # Quick function to activate python_practice environment
    ENV_NAME="python_practice"

    if ! conda env list | grep -q "^${ENV_NAME}"; then
        echo "❌ Environment '${ENV_NAME}' not found. Run ./quick_setup.sh first."
        return 1
    fi

    conda activate ${ENV_NAME}
    if [ $? -eq 0 ]; then
        echo "✅ Activated ${ENV_NAME} environment"
        echo "💡 Run 'python practice_tests.py' to start practicing"
    else
        echo "⚠️  Activation failed. Try: conda run -n ${ENV_NAME} python practice_tests.py"
    fi
}

# Usage: Just type 'practice_env' in terminal to activate the environment
