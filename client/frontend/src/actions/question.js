import axios from "axios";
import { createMessage, returnErrors } from "./messages";
import { GET_STORY, ADD_STORY } from "./types";

// ADD Question
export const addQuestion = question => (dispatch, getState) => {
    axios
        .post("/api/qa/", question, getState)
        .then(res => {
            //dispatch(createMessage({ addLead: "Lead Added" }) );
            console.log("response:", res.data.story)
            axios.get(`/api/story/${res.data.story}/`).then(res2 => {
                dispatch({
                    type: GET_STORY,
                    payload: res2.data
                });
            })

        })
        .catch(err =>
            dispatch(returnErrors(err.response.data, err.response.status))
        );
};

