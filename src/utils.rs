use crate::error::CoreMLError;

/// Validate input shape dynamically against expected shape
pub fn validate_coreml_shape(
    expected_shape: &[usize],
    actual_shape: &[usize],
    feature_name: &str,
) -> Result<(), CoreMLError> {
    if expected_shape.is_empty() && !actual_shape.is_empty() {
        return Err(CoreMLError::BadInputShape(format!(
            "Input feature name '{}' not expected!",
            feature_name
        )));
    }
    // Flexible shape matching: 0 means any dimension
    if expected_shape.len() != actual_shape.len()
        || !expected_shape
            .iter()
            .zip(actual_shape.iter())
            .all(|(&c, &a)| c == 0 || c == a)
    {
        return Err(CoreMLError::BadInputShape(format!(
            "expected shape {:?} found {:?}",
            expected_shape, actual_shape
        )));
    }
    Ok(())
}
