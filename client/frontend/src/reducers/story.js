import { GET_STORY, ADD_STORY } from "../actions/types.js";

const initialState = {
    story: {}
};

export default function (state = initialState, action) {
    switch (action.type) {
        case GET_STORY:
            return {
                ...state,
                story: action.payload
            };

        case ADD_STORY:
            return {
                ...state,
                story: action.payload
            };

        default:
            return state;
    }
}