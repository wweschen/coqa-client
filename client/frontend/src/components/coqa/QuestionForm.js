import React, { Component } from 'react'
import { connect } from "react-redux";
import PropTypes from "prop-types";
import { addQuestion } from "../../actions/question";

export class QuestionForm extends Component {
    state = {
        turn: 1,
        question: "",
        answer: "",
        story: ""
    };

    static propTypes = {
        addQuestion: PropTypes.func.isRequired
    };

    onChange = e => this.setState({ [e.target.name]: e.target.value });

    onSubmit = e => {
        e.preventDefault();

        const { turn, question, answer, story } = this.state;
        const qa = {
            turn: this.props.story.story.qa_set.length + 1,
            question,
            answer,
            story: this.props.story.story.id
        };
        this.props.addQuestion(qa);
        this.setState({
            id: "",
            question: ""
        });
    };

    render() {
        const { turn, question, answer, story } = this.state;
        return (
            <div className={[typeof this.props.story.story.id == "undefined" ? 'hidden' : ''].join(" card card-body mt-4 mb-4 ")}>
                <h2>New Question</h2>
                <form onSubmit={this.onSubmit}>
                    <div className="form-group">

                        <label>Question:</label>
                        <input
                            className="form-control"
                            type="text"
                            name="question"
                            onChange={this.onChange}
                            value={question}
                        />
                    </div>
                    <div className="float-right form-group">
                        <button type="submit" className="btn btn-primary"> Submit </button>
                    </div>
                </form>
            </div>
        )
    }
}

function mapStateToProps(state) {

    const { id, story, qa_set } = state;
    console.log("question state:", state)
    return {
        id: id,
        story: story,
        qa_set: qa_set
    }
}

export default connect(
    mapStateToProps,
    { addQuestion }
)(QuestionForm)
