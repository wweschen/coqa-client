import React, { Component } from 'react'
import { connect } from "react-redux";
import PropTypes from "prop-types";

export class StoryQADisplay extends Component {
    constructor(props) {
        super(props)
        this.state = {
            story: "",
            qa_set: []
        };
        // preserve the initial state in a new object
        this.baseState = this.state
    }

    resetStory = () => {
        this.props.dispatch({
            type: 'ADD_STORY',
            payload: this.baseState
        })
    }

    render() {

        const { id, story, qa_set } = this.state;

        return (

            <div className={[typeof this.props.story.story.id == "undefined" ? 'hidden' : ''].join(" card card-body mt-4 mb-4 ")}>
                <div className="row">
                    <div className="  row   col-md-12">
                        <div className="  col-md-6"><h2>Story</h2></div>
                        <div className=" float-right col-md-6 text-right">
                            <button type="button" onClick={this.resetStory} className="btn btn-primary "> Try a new story </button>
                        </div>
                    </div>
                </div>

                <div className="form-group">
                    <textarea
                        className="form-control"
                        type="text"
                        name="story"
                        value={this.props.story.story.story}
                    />
                </div>
                <div>
                    <table className="table table-striped">
                        <thead>
                            <tr>
                                <th >Turn</th>
                                <th>Question</th>
                                <th>Answer</th>

                            </tr>
                        </thead>
                        <tbody>
                            {this.props.story.story.qa_set.map(qa => (
                                <tr key={qa.id}>
                                    <td>{qa.turn}</td>
                                    <td>{qa.question}</td>
                                    <td>{qa.answer}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                </div>
            </div>
        )
    }
}

function mapStateToProps(state) {
    console.log("state received:", state);
    return state

};
export default connect(
    mapStateToProps
)(StoryQADisplay);
