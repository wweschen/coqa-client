import React, { Component } from 'react'
import { connect } from "react-redux";
import PropTypes from "prop-types";
import { addStory } from "../../actions/story";

export class StoryForm extends Component {
    state = {
        id: "",
        story: "",
        qa_set: []
    };

    static propTypes = {
        story: PropTypes.object,
        addStory: PropTypes.func.isRequired
    };

    onChange = e => this.setState({ [e.target.name]: e.target.value });

    onSubmit = e => {
        e.preventDefault();
        const { id, story, qa_set } = this.state;
        const storyObj = { id, story, qa_set };
        this.props.addStory(storyObj);
        this.setState({
            id: "",
            story: ""
        });
    };
    render() {
        const { id, story, qa_set } = this.state;
        return (
            <div className={[typeof this.props.story.story.id !== "undefined" ? 'hidden' : ''].join(" card card-body mt-4 mb-4 ")}>
                <h2>New Story</h2>
                <form onSubmit={this.onSubmit}>
                    <div className="form-group">
                        <label>Story Text:</label>
                        <textarea
                            className="form-control"
                            type="text"
                            name="story"
                            onChange={this.onChange}
                            value={story}
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

    return {
        id: id,
        story: story,
        qa_set: qa_set
    }
}

export default connect(
    mapStateToProps,
    { addStory }
)(StoryForm)
