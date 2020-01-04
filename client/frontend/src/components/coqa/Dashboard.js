import React, { Component, Fragment } from 'react'
import { connect } from "react-redux";
import StoryForm from './StoryForm'
import QuestionForm from './QuestionForm'
import StoryQADisplay from './StoryQADisplay'

export class Dashboard extends Component {
  render() {
    return (
      <Fragment>
        <StoryForm />
        {typeof this.props.story.story.id !== "undefined" ? <StoryQADisplay /> : null}
        <QuestionForm />
      </Fragment>
    )
  }
}
function mapStateToProps(state) {

  const { id, story, qa_set } = state;
  console.log("dashboard:", state)
  return {
    id: id,
    story: story,
    qa_set: qa_set
  }
}
export default connect(
  mapStateToProps
)(Dashboard)
