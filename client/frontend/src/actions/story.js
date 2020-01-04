import axios from "axios";
import { createMessage, returnErrors } from "./messages";
import { GET_STORY, ADD_STORY } from "./types";

// GET STORY
export const getStory = id => (dispatch, getState) => {
    axios
        .get("/api/story/${id}/", getState)
        .then(res => {
            dispatch({
                type: GET_STORY,
                payload: res.data
            });
        })
        .catch(err =>
            dispatch(returnErrors(err.response.data, err.response.status))
        );
};

// ADD STORY
export const addStory = story => (dispatch, getState) => {
    axios
        .post("/api/story/", story, getState)
        .then(res => {
            //dispatch(createMessage({ addLead: "Lead Added" }) );
            dispatch({
                type: ADD_STORY,
                payload: res.data
            });
        })
        .catch(err =>
            dispatch(returnErrors(err.response.data, err.response.status))
        );
};

