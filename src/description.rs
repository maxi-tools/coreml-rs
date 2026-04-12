use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct FeatureDescription {
    pub name: String,
    pub shape: Vec<usize>,
    pub type_name: String,
}

#[derive(Debug, Clone)]
pub struct ModelDescription {
    pub inputs: HashMap<String, FeatureDescription>,
    pub outputs: HashMap<String, FeatureDescription>,
}

impl ModelDescription {
    pub fn input_names(&self) -> Vec<String> {
        self.inputs.keys().cloned().collect()
    }

    pub fn output_names(&self) -> Vec<String> {
        self.outputs.keys().cloned().collect()
    }
}

impl From<crate::ffi::ModelDescription> for ModelDescription {
    fn from(desc: crate::ffi::ModelDescription) -> Self {
        let mut inputs = HashMap::<String, FeatureDescription>::new();
        for name in desc.inputs() {
            let shape = desc.input_shape(&name);
            inputs.insert(
                name.clone(),
                FeatureDescription {
                    name: name.clone(),
                    shape,
                    type_name: "unknown".to_string(), // type_name for inputs not exposed in ffi yet
                },
            );
        }
        let mut outputs = HashMap::<String, FeatureDescription>::new();
        for name in desc.output_names() {
            let shape = desc.output_shape(&name);
            let type_name = desc.output_type(&name);
            outputs.insert(
                name.clone(),
                FeatureDescription {
                    name: name.clone(),
                    shape,
                    type_name,
                },
            );
        }
        ModelDescription { inputs, outputs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_model_description_names() {
        let mut inputs = HashMap::new();
        inputs.insert(
            "input1".to_string(),
            FeatureDescription {
                name: "input1".to_string(),
                shape: vec![1, 224, 224, 3],
                type_name: "image".to_string(),
            },
        );
        inputs.insert(
            "input2".to_string(),
            FeatureDescription {
                name: "input2".to_string(),
                shape: vec![1, 10],
                type_name: "multiArray".to_string(),
            },
        );

        let mut outputs = HashMap::new();
        outputs.insert(
            "output1".to_string(),
            FeatureDescription {
                name: "output1".to_string(),
                shape: vec![1, 1000],
                type_name: "multiArray".to_string(),
            },
        );

        let desc = ModelDescription { inputs, outputs };

        let mut input_names = desc.input_names();
        input_names.sort();
        assert_eq!(
            input_names,
            vec!["input1".to_string(), "input2".to_string()]
        );

        let mut output_names = desc.output_names();
        output_names.sort();
        assert_eq!(output_names, vec!["output1".to_string()]);
    }
}
