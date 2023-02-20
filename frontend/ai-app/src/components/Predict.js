import React from 'react';
import {
    Form,
    TextArea,
    Button,
    Icon
} from 'semantic-ui-react';

export default function Translate() {
    return (
        <div>
            <div className="app-header">
                <h2 className="header">Texty Translator</h2>
            </div>

            <div className='app-body'>
                <div>
                    <Form>
                        <Form.Field
                            control={TextArea}
                            placeholder='Type text as input..'
                        />

                        <select className="language-select">
                            <option>Please Select Model..</option>
                        </select>

                        <Form.Field
                            control={TextArea}
                            placeholder='Your prediction..'
                        />

                        <Button
                            color="blue"
                            size="large"
                        >Predict</Button>
                    </Form>
                </div>
            </div>
        </div>
    )
}
