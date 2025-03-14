from flask import Flask, render_template, request, flash
import joblib
import pandas as pd

# Initialize the flask application
app = Flask(__name__)
app.secret_key = 'du8382hd1'  # For flash messages

# Load model components
try:
    loaded = joblib.load('loan_pipeline.pkl')
    model = loaded['model']
    class_names = loaded['class_names']

    # After loading model - Debug
    print("Model classes:", model.classes_)
    # Should output: ['0' '1']
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}") from e

# Expected features and validation rules
EXPECTED_FEATURES = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
                     'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                     'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

VALIDATION_RULES = {
    'cibil_score': (300, 900),
    'loan_term': (1, 20),
    'no_of_dependents': (0, 10)
}


# Home Page
@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        try:
            input_data = {}
            errors = []

            # Validate and parse all fields
            for field in EXPECTED_FEATURES:
                value = request.form.get(field, '')

                # Handle numeric fields
                if field in ['no_of_dependents', 'income_annum', 'loan_amount',
                             'loan_term', 'cibil_score', 'residential_assets_value',
                             'commercial_assets_value', 'luxury_assets_value',
                             'bank_asset_value']:

                    # html parses them as a string so turn them to int or float
                    input_data[field] = float(value) if '.' in value else int(value)

                    # Check validation ranges
                    if field in VALIDATION_RULES:
                        min_val, max_val = VALIDATION_RULES[field]
                        if not (min_val <= input_data[field] <= max_val):
                            errors.append(f"{field.replace('_', ' ')} must be between {min_val}-{max_val}")
                else:
                    # Handle categorical fields
                    input_data[field] = value.strip().title()

            if errors:
                for error in errors:
                    flash(error) # show the errors to the user
                return render_template('index.html', form_data=request.form)

            # Create Dataframe with proper column order
            try:
                input_df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
            except KeyError as e:
                app.logger.error(f"Missing columns: {e}")
                flash("Internal error: Feature mismatch")
                return render_template('index.html', form_data=request.form)

            # Get prediction and probability
            try:
                prediction_result = model.predict(input_df)
                prediction_idx = int(prediction_result[0])

                probability = model.predict_proba(input_df)[0].max()

                result_num = model.classes_[prediction_idx]
                if result_num == 0:
                    result = 'Approved'
                else:
                    result = 'Rejected'

            except Exception as e:
                app.logger.error(f"Prediction failed: {str(e)}")
                flash("Prediction error. Please check your inputs.")
                return render_template('index.html', form_data=request.form)

            return render_template('result.html', result=result,
                                   probability=round(probability * 100, 2), form_data=input_data)

        except Exception as e:
            app.logger.error(f"General error: {str(e)}")
            flash("An unexpected error occurred Please try again.")
            return render_template('index.html', form_data=request.form)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
