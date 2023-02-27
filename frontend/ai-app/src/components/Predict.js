import React, { useState, useEffect } from 'react';
import {
    Form,
    TextArea,
    Button,
    Icon
} from 'semantic-ui-react';
import axios from 'axios'

export default function Predict() {
    const [inputText, setInputText] = useState('');
    const [modelList, setModelList] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [resultText, setResultText] = useState('');

    useEffect(() => {
        axios.get('http://127.0.0.1:8000/models')
            .then((response) => {
            console.log(response.data)
            console.log(response.data.models)
                setModelList(response.data.models)
            })
    }, [])

    const modelAssign = (selectedModelName) => {
        setSelectedModel(selectedModelName.target.value)
    }

    const predictText = () => {
        axios.get(`http://127.0.0.1:8000/predict/${selectedModel}`,
         {params: {text :inputText} })
        .then((response) => {
            setResultText(response.data.prediction)
        })
    }
    return (
        <div>
            <div className="app-header">
                <h2 className="header">Text Prediction</h2>
            </div>

            <div className='app-body'>
                <div>
                    <Form>
                        <Form.Field
                            control={TextArea}
                            placeholder='Type text as input..'
                            onChange={(e) => setInputText(e.target.value)}
                        />

                        <select className="model-select" onChange={modelAssign}>
                            <option>Please Select Model..</option>
                            {modelList.map((model) => {
                            return (
                                <option value={model.name} key={model.name}>
                                    {model.description}
                                </option>
                            )
                            })}
                        </select>

                        <Form.Field
                            control={TextArea}
                            placeholder='Your prediction..'
                            value={resultText}
                        />

                        <Button
                            color="blue"
                            size="large"
                            onClick={predictText}
                        >Predict</Button>
                    </Form>
                </div>
            </div>
        </div>
    )
}
