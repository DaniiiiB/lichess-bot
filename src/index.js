//const fetch = require('node-fetch');
const https = require('https');
const axios = require('axios');
const EventEmitter = require('events');
const prompt = require("prompt-sync")({ sigint: true });

require('dotenv').config();

const {
    BOT_TOKEN,
    BOT_NAME,
} = process.env;

const BASE_URL = 'https://lichess.org';
const headers = {
    Authorization: `Bearer ${BOT_TOKEN}`,
};


function init(){
    listenForChallenges();
}

async function startGame(event){
    const gameId = event.game.id;
    const game = stream(`${BASE_URL}/api/bot/game/stream/${gameId}`);
    let fullGameState;
    game.on('event', async (gameEvent) => {
        if (gameEvent.type === 'gameFull') {
            fullGameState = gameEvent;
        }
        const gameState = gameEvent.type === 'gameFull' ? gameEvent.state : gameEvent;
        console.log(gameEvent);
            const allMoves = gameState.moves.split(' ').filter(e => e != '');
            const turn = allMoves.length % 2;
            if (
                (turn === 0 && fullGameState.white.id === BOT_NAME) ||
                (turn === 1 && fullGameState.black.id === BOT_NAME)
            ) {

                //console.log("BOT'S TURN");
                //makeMove(gameId, 'e2e4');
                //const move = prompt("Bot move: ")
                const fs = require('fs')
                fs.writeFile('/lichess-bot/src/Comunication/moves.txt', gameState.moves, err => {
                if (err) {
                    console.error(err)
                    return
                }
                })

                let { PythonShell } = require('python-shell');
                let options = {
                    mode: 'text',
                    scriptPath: './src'
                };
                PythonShell.run('./evaluate.py', options, function (err, results) {
                    if (err) throw err;
                    fs.readFile('/lichess-bot/src/Comunication/move.txt', function (err,data) {
                        if (err) {
                          return console.log(err);
                        }
                        makeMove(gameId, data);
                      });
                });
            }
    })
}

async function makeMove(gameId, move) {
    try {
        const { data } = await axios({
            method: 'POST',
            url: `${BASE_URL}/api/bot/game/${gameId}/move/${move}`,
            headers,
        });
        return data;
    } catch (error) {
        return {
            ok: false,
        }
    }
}

async function acceptChallenge(event){
    const { data } = await axios({
        method: 'POST',
        url: `${BASE_URL}/api/challenge/${event.challenge.id}/accept`,
        headers,
    });
}

function stream(url){
    const emitter = new EventEmitter();
    (async () => {
        const response = await axios({
            method: 'get',
            url,
            responseType: 'stream',
            headers,
        });
        for await (const chunk of response.data) {
            const jsonString = new TextDecoder('utf-8').decode(chunk, { stream: true });
            if (jsonString.trim()) {
                const parts = jsonString.split('\n');
                parts.forEach((part) => {
                    if (part.trim()) {
                        const event = JSON.parse(part);
                        emitter.emit('event',event)
                    }
                })
            }
        }
    })();
    return emitter;
}

async function listenForChallenges(){
    const challenges = stream(`${BASE_URL}/api/stream/event`);
    challenges.on('event', (event) => {
        if (event.type === 'challenge' && event.challenge.challenger.id === 'daniib55') {
            acceptChallenge(event);
            console.log(event);
        }
        if (event.type === 'gameStart') {
            startGame(event);
            console.log(event);
        }

    });
}

init();